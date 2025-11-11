Установка зависимостей:
```bash
pip install -r requirements.txt
```

Чтобы запустить обучение:
```bash
python ./src/lora_fit_and_save.py --train_data_dir ./acvarel_caption_en --output_folder ./outputs --resolution 512 --learning_rate 0.0005 --epochs 20
```

Чтобы запустить генерацию после обучения:
```bash
python ./src/generate_images.py --lora_path outputs --prompt "a big fat cat on sofa" --compare_base --seed 12345 --num_images 10
```