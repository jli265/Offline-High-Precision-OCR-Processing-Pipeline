# https://gemini.google.com/app/213cb915dec24b9f

import os
os.environ["FLAGS_use_mkldnn"] = "0"  # 禁用 OneDNN 加速，防止底层算子崩溃

import cv2
import numpy as np
import logging
from typing import List, Dict, Any
from paddleocr import PaddleOCR  # 需要 paddleocr==2.7.3 版本

# 配置日志记录
logging.basicConfig(level=logging.INFO, format="%(asctime)s - [%(levelname)s] - %(message)s")
logger = logging.getLogger("OCR_Engine")


class AdvancedOCRProcessor:
    """
    企业级高效 OCR 处理框架
    封装了图像预处理、文本检测、文本识别与后处理流水线
    """

    def __init__(self, use_gpu: bool = True, lang: str = "ch"):
        self.use_gpu = use_gpu
        self.lang = lang
        self._initialize_model()

    def _initialize_model(self) -> None:
        """初始化 PaddleOCR 检测与识别引擎"""
        logger.info(f"正在初始化 OCR 引擎 [语言: {self.lang}, GPU加速: {self.use_gpu}]...")
        try:
            # 实例化 PaddleOCR，自动下载/加载模型
            self.engine = PaddleOCR(use_angle_cls=True, lang=self.lang, use_gpu=self.use_gpu)
            logger.info("OCR 引擎初始化成功。")
        except Exception as e:
            logger.error(f"OCR 引擎初始化失败: {e}")
            raise e

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        图像预处理流水线
        - 灰度化与去噪
        - 自适应直方图均衡化（增强对比度）
        - 转回 3 通道以兼容 PaddleOCR 输入格式
        """
        if image is None or image.size == 0:
            raise ValueError("输入图像为空，请检查图像路径或数据流。")

        # 1. 灰度转换
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # 2. 噪声抑制（高斯滤波）
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)

        # 3. 对比度增强 (CLAHE)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(blurred)

        # 4. 转换回 3 通道 BGR 图像（满足 PaddleOCR 内部矩阵运算的维度要求）
        enhanced_bgr = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)

        logger.debug("图像预处理完成。")
        return enhanced_bgr

    def detect_and_recognize(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        核心执行流水线：调用 PaddleOCR 进行文本检测与识别
        """
        results = []
        try:
            # 调用真实 OCR 模型，cls=True 启用方向分类矫正
            raw_outputs = self.engine.ocr(image, cls=True)

            # 解析 PaddleOCR 的返回结构
            if raw_outputs and raw_outputs[0]:
                for line in raw_outputs[0]:
                    box = line[0]  # 文本框四个角的坐标 [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
                    text = line[1][0]  # 识别出的文字内容
                    confidence = line[1][1]  # 置信度

                    results.append({
                        "box": box,
                        "text": text,
                        "confidence": float(confidence)
                    })
            logger.info(f"文本检测与识别执行成功，共发现 {len(results)} 处文本。")
        except Exception as e:
            logger.error(f"OCR 模型推理过程发生异常: {e}")

        return results

    def post_process(self, raw_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        后处理与过滤
        - 置信度过滤
        """
        filtered_results = []
        confidence_threshold = 0.5  # 置信度阈值，可根据业务精度需求调整

        for item in raw_results:
            if item.get("confidence", 1.0) >= confidence_threshold:
                filtered_results.append(item)

        return filtered_results

    def __call__(self, image_path: str) -> List[Dict[str, Any]]:
        """
        统一调用入口（支持中文路径）
        """
        # 使用 np.fromfile + cv2.imdecode 解决 Windows 下中文路径读取失败的问题
        try:
            image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        except Exception as e:
            image = None

        if image is None or image.size == 0:
            logger.error(f"无法读取图像文件，请检查文件路径或文件是否损坏: {image_path}")
            return []

        # 1. 预处理
        processed_img = self.preprocess(image)

        # 2. 检测与识别
        raw_outputs = self.detect_and_recognize(processed_img)

        # 3. 后处理
        final_results = self.post_process(raw_outputs)

        return final_results


# ==================== 使用示例 ====================
if __name__ == "__main__":
    # 初始化 OCR 处理器（首次运行会自动下载模型，请保持网络畅通）
    ocr_pipeline = AdvancedOCRProcessor(use_gpu=True, lang="ch")

    # 替换为你实际的测试图片路径（支持绝对路径或包含中文的路径）
    文本图片地址 = r"D:\work\work5\科锐国际：苹果富士康\OCR技术\测试样本0.jpeg"

    # 执行识别任务
    results = ocr_pipeline(文本图片地址)

    # 打印最终识别结果
    print("\n" + "=" * 10 + " OCR 识别结果 " + "=" * 10)
    if not results:
        print("未检测到有效文字，请检查图片是否清晰。")
    for res in results:
        print(f"识别文本: {res['text']} | 置信度: {res['confidence']:.4f}")