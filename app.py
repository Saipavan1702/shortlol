import time
import math
# import ffmpeg
from srt_to_vtt import srt_to_vtt

import torch
from faster_whisper import WhisperModel
from flask import Flask, render_template, request, Response, jsonify,send_from_directory
import os
import torch

from diffusers import AnimateDiffPipeline, LCMScheduler, MotionAdapter
from diffusers.utils import export_to_gif
from flask_cors import CORS
adapter = MotionAdapter.from_pretrained("wangfuyun/AnimateLCM", torch_dtype=torch.float16)
pipe = AnimateDiffPipeline.from_pretrained("emilianJR/epiCRealism", motion_adapter=adapter, torch_dtype=torch.float16)
pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config, beta_schedule="linear")
 
pipe.load_lora_weights("wangfuyun/AnimateLCM", weight_name="AnimateLCM_sd15_t2v_lora.safetensors", adapter_name="lcm-lora")
pipe.set_adapters(["lcm-lora"], [0.8])
pipe.enable_vae_slicing()
pipe.enable_model_cpu_offload()
 
 
 
def video_generator(prompt):
 
  output = pipe(
      prompt=prompt,
 
  # prompt="sad boy sitting on bench, bald, face, half body, body, high detailed skin, skin pores, coastline, overcast weather, wind, waves, 8k uhd, dslr, soft lighting, high quality, film grain, Fujifilm XT3",
  negative_prompt=  "anime",
  num_frames=20,
  guidance_scale=2.0,
  num_inference_steps=16,
  generator=torch.Generator("cuda").manual_seed(0),
  )
  frames = output.frames[0]
  torch.cuda.empty_cache()
 
  export_to_gif(frames, "anime_outputd/animatelcm.gif")
  path="animatelcm.gif"
  return path
# from whisper import WhisperModel
 
def transcribe(audio):
    model = WhisperModel("small")
    segments, info = model.transcribe(audio)
    language = info[0]
    print("Transcription language", language)
   
    # Prepare to create SRT content
    srt_content = ""
   
    for index, segment in enumerate(segments):
        start_time = segment.start
        end_time = segment.end
        text = segment.text
 
        # Format times into SRT timestamp format
        start = format_srt_timestamp(start_time)
        end = format_srt_timestamp(end_time)
       
        srt_content += f"{index + 1}\n"
        srt_content += f"{start} --> {end}\n"
        srt_content += f"{text}\n\n"
 
    # Save to SRT file
    srt_filename = "anime_outputd/subtitles.srt"
    with open(srt_filename, 'w') as file:
        file.write(srt_content)
    
    print(f"Subtitles saved to {srt_filename}")
    return language, segments
 
def format_srt_timestamp(seconds):
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    milliseconds = int((seconds - int(seconds)) * 1000)
    return f"{int(hours):02}:{int(minutes):02}:{int(seconds):02},{milliseconds:03}"
 
# Example usage
# audio_file = 'your_audio_file.mp3'  # Replace with your audio file path
# transcribe(audio_file)
 
from concurrent.futures import ThreadPoolExecutor
 
 
def thread_three(prompt,audio_path):
    def prediction_surface_parallel(num,prompt):
        if num==1:
            prediction_output=[]
            prediction=video_generator(prompt)
            prediction_output.append("prediction_grade")
            return prediction_output
        else:
            surface_output=[]
            language, segments=transcribe(audio_path)
            # surface_output.append("Surace_grade")
            surface_output.append(segments)
            # series,a,b,c=get_series(image_path)
            return surface_output
    num=[1,2]
    op=[]
    # image_path=image_path
    image_path1=[str(prompt),str(audio_path)]
    with ThreadPoolExecutor(max_workers=16) as executor:
        for a in executor.map(prediction_surface_parallel,num,image_path1):
            op.append((a))
            print('done')
    return op


app = Flask(__name__)
CORS(app)


import logging

# Create a logger
logger = logging.getLogger(__name__)

# Set the logging level to capture only ERROR and above
logger.setLevel(logging.ERROR)

# Create a file handler
file_handler = logging.FileHandler('error_back.log')

# Create a formatter and set it to the file handler
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)

# Add the file handler to the logger
logger.addHandler(file_handler)


UPLOAD_FOLDER = 'audio_files'
OUTPUT_FOLDER = 'anime_outputd'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/back', methods=['POST'])
def back_card():
    if 'audio_file' not in request.files:
        return Response('Missing file, please check!', status=404)
    
    uploaded_file = request.files['audio_file']
    uid = request.form.get('prompt', None)
    
    file_name = uploaded_file.filename.replace(" ", "_")
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file_name)
    uploaded_file.save(file_path)

    audio_path = f"{app.config['UPLOAD_FOLDER']}/{file_name}"
    
    # Replace `thread_three` with the actual processing function
    op = thread_three(uid, audio_path)
    path_to_converted_vtt_file="anime_outputd/subtitles.vtt"
    srt_to_vtt("anime_outputd/subtitles.srt", path_to_converted_vtt_file)
    gif_link = f"http://164.52.196.127:5001/animatelcm.gif"
    srt_link = f"http://164.52.196.127:5001/subtitles.srt"
    vtt_link = f"http://164.52.196.127:5001/subtitles.vtt"
    audio_url = f"http://164.52.196.127:5001/{app.config['UPLOAD_FOLDER']}/{file_name}"
    data1 = {'gif_link': gif_link, 'srt_link': srt_link, 'audio_url': audio_url,'vtt_link':vtt_link}

    return jsonify(data1), 200

@app.route('/<filename>')
def serve_file(filename):
    return send_from_directory(OUTPUT_FOLDER, filename)

@app.route('/audio_files/<filename>')
def serve_audio_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=False, port=5001)
