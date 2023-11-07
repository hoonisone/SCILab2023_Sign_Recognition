from PaddleOCR import paddleocr
import json



def save_result(predict, image_path, root_dir):
    image_file_name = image_path.split("\\")[-1].split(".")[0]
    file_path = f"{root_dir}/predicts/predict_({image_file_name}).json"

    with open(file_path, "w") as f:
        f.write(json.dumps(predict))
    
def main():
    root_dir = "test1"
    image_dir = f"{root_dir}/images"

    image_paths, predicts = paddleocr.main(image_dir)
    # PADDLEOCR에서 제공하는 코드이다. 설정된 디렉터리 안에 모든 이미지에 대해 추론을 한다.
    # 결과값은 반환되도록 수정함

    for predict, image_path in zip(predicts, image_paths):
        save_result(predict, image_path, root_dir)

if __name__=="__main__":
    main()

