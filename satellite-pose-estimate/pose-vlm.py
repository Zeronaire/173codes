import json
import os
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import VideoLlavaProcessor, VideoLlavaForConditionalGeneration, BitsAndBytesConfig, Trainer, TrainingArguments
from transformers.trainer_utils import get_last_checkpoint
import torch
from moviepy.editor import ImageSequenceClip  # For converting image sequences to video
from scipy.spatial.transform import Rotation  # For quaternion to Euler conversion
import numpy as np  # For velocity computation
import av  # For video loading
from peft import prepare_model_for_kbit_training

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def read_video_pyav(container, indices):
    '''
    Decode the video with PyAV decoder.
    Args:
        container (`av.container.input.InputContainer`): PyAV container.
        indices (`List[int]`): List of frame indices to decode.
    Returns:
        result (np.ndarray): np array of decoded frames of shape (num_frames, height, width, 3).
    '''
    frames = []
    container.seek(0)
    start_index = indices[0]
    end_index = indices[-1]
    for i, frame in enumerate(container.decode(video=0)):
        if i > end_index:
            break
        if i >= start_index and i in indices:
            frames.append(frame)
    return np.stack([x.to_ndarray(format="rgb24") for x in frames])


# New function to prepare MSPD dataset
def prepare_mspd_dataset(mspd_dir, output_json_path, dt=1.0, fps=1, modality='RGB', chunk_size=4):
    """
    Prepares MSPD dataset for Video-LLaVA training.
    - mspd_dir: Root directory of extracted MSPD dataset.
    - output_json_path: Path to save the output JSON in conversation format.
    - dt: Time delta between frames in seconds (for velocity computation if not provided).
    - fps: Frames per second for generated MP4 videos.
    - modality: Image modality to use (e.g., 'RGB', 'IR', 'Depth').
    - chunk_size: Number of frames per sub-sequence (default 5).
    
    Assumes structure:
    - mspd_dir/sequences/seq001/{modality}/frame_0001.png, frame_0002.png, ...
    - mspd_dir/sequences/seq001/labels.json: List of dicts with 'frame_id', 'translation' (list[3]), 'quaternion' (list[4]), 'timestamp' (optional).
    
    Computes angular velocity if not in labels using quaternion log difference.
    Converts quaternions to Euler angles (roll, pitch, yaw in degrees).
    Generates MP4 video per sub-sequence (chunks of chunk_size frames).
    """
    data = []
    seq_dirs = [os.path.join(mspd_dir, 'sequences', d) for d in os.listdir(os.path.join(mspd_dir, 'sequences')) if os.path.isdir(os.path.join(mspd_dir, 'sequences', d))]
    
    for seq_dir in seq_dirs:
        seq_id = os.path.basename(seq_dir)
        frame_dir = os.path.join(seq_dir, modality)
        labels_path = os.path.join(seq_dir, 'labels.json')
        
        if not os.path.exists(labels_path) or not os.path.exists(frame_dir):
            print(f"Skipping {seq_id}: Missing labels or frames.")
            continue
        
        with open(labels_path, 'r') as f:
            labels = json.load(f)
        
        # Sort frames by name
        #frames = sorted([os.path.join(frame_dir, f) for f in os.listdir(frame_dir) if f.endswith(('.png', '.jpg'))], key=lambda x: int(os.path.basename(x).split('_')[1].split('.')[0]))
        frames = sorted([os.path.join(frame_dir, f) for f in os.listdir(frame_dir) if f.endswith(('.png', '.jpg'))], key=lambda x: int(os.path.basename(x).split('.')[0]))

        if len(frames) != len(labels):
            print(f"Skipping {seq_id}: Frame-label mismatch.")
            continue
        
        # Split into chunks of chunk_size frames
        for chunk_idx, start in enumerate(range(0, len(labels), chunk_size)):
            sub_labels = labels[start:start + chunk_size]
            sub_frames = frames[start:start + chunk_size]
            if len(sub_labels) == 0:
                continue  # Skip empty chunks
            if len(sub_labels) < chunk_size:
                continue  # Skip incomplete chunks to ensure uniform size
            
            sub_seq_id = f'{seq_id}_chunk{chunk_idx}'
            video_path = os.path.join(seq_dir, f'{sub_seq_id}.mp4')
            clip = ImageSequenceClip(sub_frames, fps=fps)
            clip.write_videofile(video_path, codec='libx264')
            
            # Process sub_labels: Convert to Euler, compute velocities
            prev_rot = None
            conversations = []
            for i, label in enumerate(sub_labels):
                # Quaternion to Euler (ZYX convention, degrees)
                rot = Rotation.from_quat(label['quaternion'])
                euler = rot.as_euler('zyx', degrees=True)  # [yaw, pitch, roll]
                
                # Angular velocity (if not provided)
                if 'velocity' in label:
                    velocity = label['velocity']
                else:
                    if prev_rot is not None:
                        # Quaternion difference: log map for angular velocity
                        delta_rot = rot * prev_rot.inv()
                        log_delta = delta_rot.as_rotvec()  # rad
                        velocity = log_delta / dt  # rad/s
                        velocity = np.degrees(velocity).tolist()  # to deg/s
                    else:
                        velocity = [0.0, 0.0, 0.0]
                    prev_rot = rot
                
                gt_response = {
                    'angles': {'roll': euler[2], 'pitch': euler[1], 'yaw': euler[0]},
                    'velocity': {'wx': velocity[0], 'wy': velocity[1], 'wz': velocity[2]}
                }
                
                # Per-sub-sequence conversation (GT for the last frame in the chunk)
                if i == len(sub_labels) - 1:  # Add at end
                    conversations.append({
                        'from': 'human',
                        'value': '<video>Estimate the satellite\'s Euler angles (roll, pitch, yaw in degrees) and angular velocity (wx, wy, wz in deg/s) from this sequence.'
                    })
                    conversations.append({
                        'from': 'gpt',
                        'value': json.dumps(gt_response)  # GT for last frame
                    })
            
            data.append({
                'id': sub_seq_id,
                'video_path': video_path,
                'conversations': conversations
            })
    
    with open(output_json_path, 'w') as f:
        json.dump(data, f, indent=4)
    print(f"Prepared {len(data)} sequences in {output_json_path}")

# Step 1: Prepare your custom dataset

def load_custom_dataset(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # If paths are to frame directories, convert to video (already handled in prepare_mspd_dataset)
    
    return Dataset.from_list(data)

#prepare_mspd_dataset("mspd_data", "mspd_data.json")
# Then use "mspd_data.json" in load_custom_dataset

# Step 2: Load model and processor
model_id = "LanguageBind/Video-LLaVA-7B-hf"  # Use the HF-compatible repo
processor = VideoLlavaProcessor.from_pretrained(model_id)

# Add special tokens if not present (for video placeholder)
tokenizer = processor.tokenizer
special_tokens = {"additional_special_tokens": ["<video>"]}
num_added = tokenizer.add_special_tokens(special_tokens)

# Quantization for lower VRAM
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

model = VideoLlavaForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    #quantization_config=quantization_config,
    #device_map="auto",
    #device_map=None,
    #device_map={'':torch.cuda.current_device()},
    #device_map={'':torch.cuda.current_device()}
    #trust_remote_code=True

)

#model.gradient_checkpointing_enable()  # Enable to save VRAM

# Tie weights as recommended
#model.tie_weights()

# Resize embeddings if tokens were added
if num_added > 0:
    model.resize_token_embeddings(len(tokenizer))

#model = prepare_model_for_kbit_training(model)

# Step 3: Apply LoRA
lora_config = LoraConfig(
    r=16,  # Rank
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],  # Adapt based on model architecture
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)

# Step 4: Preprocess function for dataset
def preprocess_function(examples):
    texts = []
    videos_loaded = []
    for conv, video_path in zip(examples['conversations'], examples['video_path']):
        prompt = conv[0]['value']
        response = conv[1]['value']
        full_text = f"{prompt} {response}"  # Concat for LM training
        texts.append(full_text)
        container = av.open(video_path)
        total_frames = container.streams.video[0].frames
        indices = np.arange(0, total_frames, 1).astype(int)
        clip = read_video_pyav(container, indices)
        videos_loaded.append(clip)
    
    inputs = processor(text=texts, videos=videos_loaded, padding=True, return_tensors="pt")
    inputs['labels'] = inputs['input_ids'].clone()  # For causal LM
    return inputs

# Load and preprocess dataset
dataset = load_custom_dataset("mspd_data.json")  # Updated to use the prepared JSON
tokenized_dataset = dataset.map(preprocess_function, batched=True)


# Skip split if dataset is too small; use full for train, no eval
if len(tokenized_dataset) < 2:
    train_dataset = tokenized_dataset
    eval_dataset = None
else:
    train_test = tokenized_dataset.train_test_split(test_size=0.1)
    train_dataset = train_test['train']
    eval_dataset = train_test['test']

# Step 5: Training arguments
training_args = TrainingArguments(
    output_dir="./video_llava_finetuned",
    num_train_epochs=100,
    per_device_train_batch_size=1,  # Adjust based on VRAM
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=1,
    learning_rate=1e-5,
    weight_decay=0.01,
    warmup_steps=100,
    fp16=True,
    max_grad_norm=1.0,
    save_steps=500,
    evaluation_strategy="no" if eval_dataset is None else "steps",
    eval_steps=500,
    logging_steps=10,
    load_best_model_at_end=True if eval_dataset is None else False,
    metric_for_best_model="eval_loss" if eval_dataset is not None else None,
    report_to="none",
    #remove_unused_columns=True,
    label_names=["labels"],
    lr_scheduler_type="cosine",
    gradient_checkpointing=False,
    #deepspeed="deepspeed_config.json"
    ddp_find_unused_parameters=True
)

# Step 6: Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=processor.tokenizer,
)

#print(f"Train dataset size: {len(train_dataset)}")  # Should be ~18
#model.print_trainable_parameters()  # Check % trainable >0 (e.g., 0.05% for LoRA)

# Resume from checkpoint if exists
if get_last_checkpoint(training_args.output_dir) is not None:
    trainer.train(resume_from_checkpoint=True)
else:
    trainer.train()

#trainer.train()

# Step 7: Save the fine-tuned model
trainer.save_model("./video_llava_finetuned/final")
processor.save_pretrained("./video_llava_finetuned/final")

# Inference example (after fine-tuning)
def infer(video_path, prompt):
    container = av.open(video_path)
    total_frames = container.streams.video[0].frames
    indices = np.arange(0, total_frames, 1).astype(int)
    clip = read_video_pyav(container, indices)
    inputs = processor(text=prompt, videos=clip, return_tensors="pt").to("cuda")
    outputs = model.generate(**inputs, max_new_tokens=200)
    return processor.decode(outputs[0], skip_special_tokens=True)

#infer("mspd_data/sequences/seq001/seq001_chunk19.mp4", "<video>Estimate the satellite's Euler angles and angular velocity from this sequence.")
