Чтобы запустить генерацию базовой моделью до обучения:
```bash
python ./src/generate_base.py --prompt a big fat cat on sofa" --num-images 10
```

Чтобы запустить обучение:
```bash
python ./src/lora_fit_and_save.py --train_data_dir ./acvarel_caption --output_folder ./outputs --resolution 512 --batch_size 2 --epochs 10
```

Чтобы запустить генерацию после обучения:
```bash
python ./src/generate_images.py --lora_path outputs --prompt "a big fat cat on sofa" --num_images 10
```