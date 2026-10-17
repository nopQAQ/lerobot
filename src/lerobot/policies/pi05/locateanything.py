"""Frozen LocateAnything fused image features, adapted from starVLA-single.

The LIBERO object extraction, prompt, visual-feature injection and hidden-state
selection intentionally match its encoder wrapper. No bbox generation is needed.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

import torch
from PIL import Image
from torch import nn

_ARTICLE = r"(?:the |a |an )?"
_SENTENCE_END = r"(?:\s+and\s+close\s+it)?(?:[.!?]|$)"
_LOCATEANYTHING_INSTRUCTION_DOMAINS = frozenset({"libero"})


def _clean_object_phrase(phrase: str) -> str:
    phrase = phrase.strip().rstrip(".!?")
    phrase = re.sub(r"^(?:the|a|an)\s+", "", phrase, flags=re.IGNORECASE)
    phrase = re.sub(r"\s+and\s+close\s+it$", "", phrase, flags=re.IGNORECASE)
    return phrase.strip()


def extract_libero_objects(text: str) -> list[str]:
    """Extract object phrases from LIBERO task language.

    This intentionally covers only the compact LIBERO train-language patterns,
    not the broader LIBERO-plus paraphrase space.
    """
    text = text.strip()
    patterns: tuple[tuple[str, bool], ...] = (
        # put the white mug on the left plate and put the yellow mug on the right plate
        (
            rf"put {_ARTICLE}(.+?) (?:on|in(?:to)?|inside) {_ARTICLE}(.+?) and "
            rf"put {_ARTICLE}(.+?) (?:on|in(?:to)?|inside|to the right of) {_ARTICLE}(.+?){_SENTENCE_END}",
            False,
        ),
        # put both the alphabet soup and the cream cheese box in the basket
        (
            rf"put both {_ARTICLE}(.+?) and {_ARTICLE}(.+?) "
            rf"(?:on|in(?:to)?|inside) {_ARTICLE}(.+?){_SENTENCE_END}",
            False,
        ),
        # put both moka pots on the stove
        (
            rf"put both {_ARTICLE}(.+?) (?:on|in(?:to)?|inside) {_ARTICLE}(.+?){_SENTENCE_END}",
            False,
        ),
        # turn on the stove and put the moka pot on it -> [moka pot, stove]
        (
            rf"turn on {_ARTICLE}(.+?) and put {_ARTICLE}(.+?) "
            rf"(?:on|in(?:to)?|inside) it{_SENTENCE_END}",
            True,
        ),
        # pick up the book and place it in the caddy
        (
            rf"pick(?:\s+up)? {_ARTICLE}(.+?) and place it "
            rf"(?:on|in(?:to)?|inside) {_ARTICLE}(.+?){_SENTENCE_END}",
            False,
        ),
        # put the black bowl in the bottom drawer of the cabinet
        (
            rf"put {_ARTICLE}(.+?) (?:on|in(?:to)?|inside) {_ARTICLE}(.+?){_SENTENCE_END}",
            False,
        ),
        # put the chocolate pudding to the right of the plate
        (
            rf"put {_ARTICLE}(.+?) to the right of {_ARTICLE}(.+?){_SENTENCE_END}",
            False,
        ),
    )
    for pattern, swap_first_two in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if not match:
            continue
        objects = [_clean_object_phrase(group) for group in match.groups()]
        objects = [obj for obj in objects if obj]
        if swap_first_two and len(objects) >= 2:
            objects[0], objects[1] = objects[1], objects[0]
        # Preserve order while removing duplicates.
        deduped = list(dict.fromkeys(objects))
        if deduped:
            return deduped
    return [_clean_object_phrase(text)]


def format_locateanything_grounding_text(text: str) -> str:
    return "</c>".join(extract_libero_objects(text))


def format_locateanything_prompt(
    text: str,
    template: str,
    *,
    instruction_domain: str = "libero",
) -> str:
    instruction_domain = str(instruction_domain).strip().lower()
    if instruction_domain == "libero":
        grounding_text = format_locateanything_grounding_text(text)
    else:
        raise ValueError(
            "Unsupported LocateAnything instruction_domain "
            f"{instruction_domain!r}; expected one of "
            f"{sorted(_LOCATEANYTHING_INSTRUCTION_DOMAINS)}"
        )
    return template.format(text=grounding_text)


class LocateAnythingBackbone(nn.Module):
    """Frozen LocateAnything feature extractor for fused image-token states."""

    def __init__(
        self,
        model: nn.Module,
        processor,
        *,
        prompt_template: str,
        instruction_domain: str = "libero",
    ) -> None:
        super().__init__()
        self.model = model
        self.processor = processor
        self.prompt_template = prompt_template
        self.instruction_domain = str(instruction_domain).strip().lower()
        if self.instruction_domain not in _LOCATEANYTHING_INSTRUCTION_DOMAINS:
            raise ValueError(
                "Unsupported LocateAnything instruction_domain "
                f"{self.instruction_domain!r}; expected one of "
                f"{sorted(_LOCATEANYTHING_INSTRUCTION_DOMAINS)}"
            )
        self.image_token_index = int(model.config.image_token_index)
        self.embed_dim = int(model.config.text_config.hidden_size)
        self.num_hidden_layers = int(model.config.text_config.num_hidden_layers)
        self.merge_kernel_size = tuple(int(v) for v in model.config.vision_config.merge_kernel_size)

    def _build_messages(self, images: Sequence[Image.Image], texts: Sequence[str]):
        messages = []
        for image, text in zip(images, texts, strict=True):
            question = format_locateanything_prompt(
                text,
                self.prompt_template,
                instruction_domain=self.instruction_domain,
            )
            messages.append(
                [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": image},
                            {"type": "text", "text": question},
                        ],
                    }
                ]
            )
        return messages

    def _build_inputs(self, images: Sequence[Image.Image], texts: Sequence[str]):
        messages = self._build_messages(images, texts)
        rendered_text = [
            self.processor.py_apply_chat_template(
                message,
                tokenize=False,
                add_generation_prompt=True,
            )
            for message in messages
        ]
        flat_images = [image.convert("RGB") for image in images]
        inputs = self.processor(
            text=rendered_text,
            images=flat_images,
            videos=None,
            return_tensors="pt",
            padding=True,
        )
        device = next(self.model.parameters()).device
        return inputs.to(device)

    def scene_token_len(self, images: Sequence[Image.Image]) -> int:
        image_inputs = self.processor.image_processor(images=list(images), return_tensors="pt")
        grid_hws = image_inputs["image_grid_hws"]
        lengths = [
            int(h) * int(w) // (self.merge_kernel_size[0] * self.merge_kernel_size[1]) for h, w in grid_hws
        ]
        if not lengths:
            raise RuntimeError("LocateAnything received no images for scene-token sizing")
        first_len = lengths[0]
        if any(length != first_len for length in lengths):
            raise RuntimeError(
                "LocateAnything alignment requires equal main-view token counts "
                f"within a batch, got {lengths}"
            )
        return first_len

    @staticmethod
    def _language_model_forward(model: nn.Module):
        if getattr(model, "use_llm_lora", False):
            return model.language_model.model.model.forward
        return model.language_model.model.forward

    def _forward_language_model(self, inputs):
        pixel_values = inputs["pixel_values"].to(dtype=next(self.model.parameters()).dtype)
        input_ids = inputs["input_ids"]
        attention_mask = inputs.get("attention_mask", None)
        image_grid_hws = inputs.get("image_grid_hws", None)
        if image_grid_hws is not None and not isinstance(image_grid_hws, torch.Tensor):
            image_grid_hws = torch.from_numpy(image_grid_hws).to(
                device=pixel_values.device,
                dtype=torch.int32,
            )

        vit_embeds = self.model.extract_feature(pixel_values, image_grid_hws)
        vit_embeds = torch.cat(vit_embeds, dim=0)
        vit_embeds = self.model.mlp1(vit_embeds)

        selected = input_ids.reshape(-1) == self.image_token_index
        n_token = int(selected.sum().item())
        n_embed = int(vit_embeds.shape[0])
        if n_token != n_embed:
            raise RuntimeError(
                "LocateAnything image token/feature mismatch: "
                f"image_tokens={n_token}, image_features={n_embed}"
            )

        return self._language_model_forward(self.model)(
            input_ids=input_ids,
            visual_features=vit_embeds,
            image_token_index=self.image_token_index,
            attention_mask=attention_mask,
            use_cache=False,
            output_attentions=False,
            output_hidden_states=True,
            return_dict=True,
        )

    @staticmethod
    def _gather_image_token_embeddings(
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        image_token_index: int,
        scene_token_len: int,
    ) -> torch.Tensor:
        batch_size, seq_len, hidden_dim = hidden_states.shape
        mask = input_ids == image_token_index
        counts = mask.sum(dim=1)
        if (counts < scene_token_len).any():
            insufficient = (counts < scene_token_len).nonzero(as_tuple=False).flatten().tolist()
            raise RuntimeError(
                "LocateAnything samples have insufficient image tokens "
                f"(< {scene_token_len}): {insufficient} | counts={counts.tolist()}"
            )

        idx = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, seq_len)
        masked_pos = torch.where(mask, idx, torch.full_like(idx, -1))
        topk_pos = masked_pos.topk(k=scene_token_len, dim=-1).values
        selected_pos = topk_pos.sort(dim=-1).values
        gather_index = selected_pos.unsqueeze(-1).expand(-1, -1, hidden_dim)
        return hidden_states.gather(dim=1, index=gather_index)

    @torch.no_grad()
    def forward(
        self,
        images: Sequence[Image.Image],
        texts: Sequence[str],
    ) -> list[torch.Tensor]:
        inputs = self._build_inputs(images, texts)
        scene_len = self.scene_token_len(images)
        with torch.autocast(
            next(self.model.parameters()).device.type,
            dtype=next(self.model.parameters()).dtype,
            enabled=next(self.model.parameters()).device.type == "cuda",
        ):
            outputs = self._forward_language_model(inputs)

        return [
            self._gather_image_token_embeddings(
                hidden_states=layer_hidden,
                input_ids=inputs["input_ids"],
                image_token_index=self.image_token_index,
                scene_token_len=scene_len,
            ).contiguous()
            for layer_hidden in outputs.hidden_states[1:]
        ]
