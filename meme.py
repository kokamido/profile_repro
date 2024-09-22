import os
import json
import tempfile
import torch
import nemo.collections.asr as nemo_asr
from torch.profiler import profile, ProfilerActivity


def get_model_and_data(
    model_name:str ,
    paths2audio_files:str='audios',
    data_size=1,
    batch_size: int = 1,
    num_workers: int = 1,
    channel_selector=None,
    device="cuda",
) -> tuple[nemo_asr.models.EncDecCTCModelBPE, torch.utils.data.DataLoader]:
    print(paths2audio_files)
    asr_model = nemo_asr.models.EncDecCTCModelBPE.from_pretrained(
        model_name=model_name,
    )
    asr_model = asr_model.to(device)
    asr_model = asr_model.eval()
    with tempfile.TemporaryDirectory() as tmpdir:
        with open(os.path.join(tmpdir, "manifest.json"), "w", encoding="utf-8") as fp:
            print("aa", os.getcwd(), paths2audio_files)
            files = os.listdir(paths2audio_files)
            files = files*int(data_size // len(files)) + files[:data_size % len(files)]
            for audio_file in files:
                entry = {
                    "audio_filepath": os.path.abspath(
                        f"{paths2audio_files}/{audio_file}"
                    ),
                    "duration": 100000,
                    "text": "",
                }
                fp.write(json.dumps(entry) + "\n")

        config = {
            "paths2audio_files": paths2audio_files,
            "batch_size": batch_size,
            "temp_dir": tmpdir,
            "num_workers": num_workers,
            "channel_selector": channel_selector,
            "device":device
        }
        temporary_datalayer = asr_model._setup_transcribe_dataloader(config)
        data = []
        for batch in temporary_datalayer:
            input_signal = batch[0].to(device)
            input_signal_length=batch[1].to(device)
            processed_signal, processed_signal_length = asr_model.preprocessor(
                input_signal=input_signal, length=input_signal_length,
            )
            data.append((processed_signal, processed_signal_length))
        return asr_model, data

asr_model, data = get_model_and_data(model_name="stt_ru_conformer_ctc_large", data_size=64, device='cuda')

def go(model, data):
    with torch.no_grad():
        for processed_signal, processed_signal_length in data:
            model.encoder(audio_signal=processed_signal, length=processed_signal_length)

with profile(with_stack=True, activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    go(asr_model, data)

with open('gpu_forward_profiling', mode='w') as out:
    out.write(prof.key_averages().table(sort_by='self_cuda_time_total',max_name_column_width=4096, max_shapes_column_width=4096))