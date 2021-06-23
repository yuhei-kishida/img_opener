import cv2
from pathlib import Path
from logging import getLogger

logger = getLogger(__name__)

dir = Path(__file__).parent
png_dir = dir / "image"


def recognize_face(image: str) -> None:
    """顔を認識する

    Args:
        image (str): 顔認識させたい画像
    """
    face_cascade: Path = dir / "haarcascades" / "haarcascade_frontalface_default.xml"
    color = (0, 255, 0)
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cascade = cv2.CascadeClassifier(str(face_cascade))
    facerect = cascade.detectMultiScale(image_gray, minSize=(30, 30))
    if len(facerect) > 0:
        for rect in facerect:
            cv2.rectangle(
                image,
                tuple(rect[0:2]),
                tuple(rect[0:2] + rect[2:4]),
                color,
                thickness=2,
            )


def main():
    while True:
        order = str(input("画像番号:"))
        if not order:
            logger.warning("処理を終了します")
            break
        if order == "1":
            image: Path = png_dir / "power1.jpg"
            img = cv2.imread(str(image))
            cv2.imshow("power1", img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        # elif order == "": で増やせる
        else:
            logger.warning("画像番号を入力してください")
            continue


if __name__ == "__main__":
    main()
