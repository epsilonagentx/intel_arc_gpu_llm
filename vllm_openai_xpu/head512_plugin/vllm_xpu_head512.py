"""vLLM plugin: let the XPU flash-attention backend take head size 512.

vLLM only accepts head sizes above 256 for FLASH_ATTN when FlashAttention 4 is
available, and FA4 is CUDA-only. Intel's vllm-xpu-kernels in this image ship
compiled head-512 kernels (prefill `chunk_policy_head512`, decode
`paged_decode_kernel_template_q8_h512_p64`), so on XPU this allows exactly 512.

On its own it changes nothing: gemma-4 still gets TRITON_ATTN unless the server
is started with `--attention-backend=FLASH_ATTN`. See ../GEMMA_4_26B_A4B.md,
"Intel attention for gemma-4".
"""

from vllm.logger import init_logger

logger = init_logger("vllm.plugins.xpu_head512")


def register() -> None:
    from vllm.platforms import current_platform

    if not current_platform.is_xpu():
        return

    from vllm.v1.attention.backends.flash_attn import FlashAttentionBackend

    original = FlashAttentionBackend.supports_head_size.__func__
    if getattr(original, "_xpu_head512", False):
        return  # already applied in this process

    def supports_head_size(cls, head_size: int) -> bool:
        return head_size == 512 or original(cls, head_size)

    supports_head_size._xpu_head512 = True
    FlashAttentionBackend.supports_head_size = classmethod(supports_head_size)
    logger.info("xpu_head512: FLASH_ATTN on XPU now accepts head size 512")
