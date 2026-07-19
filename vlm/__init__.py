"""与业务无关的统一视觉语言模型 SDK。"""

from vlm.core import OpenaiApiConfig, VlmClientFactory, VlmJsonResultParser, VlmRunner


def create_vlm(config: OpenaiApiConfig) -> VlmRunner:
    """使用代码配置快速创建 VLM 运行器。"""
    return VlmRunner(VlmClientFactory.create(config), VlmJsonResultParser())


__all__ = ["OpenaiApiConfig", "VlmClientFactory", "VlmJsonResultParser", "VlmRunner", "create_vlm"]
