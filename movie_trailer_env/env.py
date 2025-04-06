# Movie Trailer Selection Using RL Agent and LongVU

# 1. **Introduction:**
#    In this notebook, we will create a reinforcement learning (RL) agent that interacts with the LongVU model to select key scenes from a video to be included in a movie trailer. 
#    The agent will choose scenes based on prompts it dynamically generates. It will receive rewards based on how well the selected scenes fit for a trailer (e.g., emotional, action-packed, or plot-relevant scenes).
#    We will define the environment, the agent's state and action spaces, the reward function, and then interact with the LongVU model to generate responses.

# 2. **Import Necessary Libraries:**

import numpy as np
import torch
import gym
from longvu.builder import load_pretrained_model
from longvu.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from longvu.conversation import conv_templates, SeparatorStyle
from longvu.mm_datautils import KeywordsStoppingCriteria, process_images, tokenizer_image_token
from decord import cpu, VideoReader

# 3. **Defining the Environment (MovieTrailerEnv)**:
#    - The environment will use the video frames as the state and select one frame at a time.
#    - The agent will generate a dynamic prompt based on the scene it chooses.
#    - It will interact with the LongVU model to determine if the scene is suitable for the trailer.
#    - A reward will be given based on the quality of the scene (action, emotional, or important).

class MovieTrailerEnv(gym.Env):
    def __init__(self, video_path):
        super(MovieTrailerEnv, self).__init__()
        
        # Initialize model and tokenizer from LongVU
        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(
            "./checkpoints/longvu_qwen", None, "cambrian_qwen",
        )
        self.model.eval()
        
        # Initialize video reader and process the video frames
        self.video_path = video_path
        self.vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
        self.fps = float(self.vr.get_avg_fps())
        self.frame_indices = np.array([i for i in range(0, len(self.vr), round(self.fps))])  # Key frames
        
        # Process the video frames and create a stack of frames
        self.video = []
        for frame_index in self.frame_indices:
            img = self.vr[frame_index].asnumpy()
            self.video.append(img)
        self.video = np.stack(self.video)
        
        # Define the action and observation spaces
        self.action_space = gym.spaces.Discrete(len(self.frame_indices))  # Action is selecting a frame
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(len(self.video), *self.video[0].shape), dtype=np.uint8)
    
    def reset(self):
        return self.video
    
    def generate_dynamic_prompt(self, scene):
        """
        This method generates dynamic prompts based on the type of scene.
        The prompt guides LongVU to analyze the scene and determine if it fits for the trailer.
        """
        if "action" in scene.lower():
            prompt = "Does this scene contain important action for a movie trailer?"
        elif "emotional" in scene.lower():
            prompt = "Does this scene have emotional intensity suitable for a movie trailer?"
        elif "cliffhanger" in scene.lower():
            prompt = "Is this scene a good cliffhanger for a movie trailer?"
        else:
            prompt = "Is this scene important for the overall story?"
        return prompt
    
    def step(self, action):
        """
        Each step consists of selecting a frame from the video, generating a prompt,
        and querying LongVU for a response. The agent will be rewarded based on whether the scene is 
        suitable for a trailer or not.
        """
        # Select the frame/scene corresponding to the action
        scene_frame = self.vr[self.frame_indices[action]].asnumpy()
        scene = "action"  # This would be dynamically detected in a real scenario
        
        # Dynamically generate a prompt for LongVU based on the scene
        prompt = self.generate_dynamic_prompt(scene)
        
        # Prepare conversation prompt for LongVU
        qs = DEFAULT_IMAGE_TOKEN + "\n" + prompt
        conv = conv_templates["qwen"].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        
        # Tokenize and prepare the input for the model
        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(self.model.device)
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)
        
        # Generate response using LongVU
        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=[scene_frame],
                image_sizes=[scene_frame.shape[:2]],  # Assuming one frame per step
                do_sample=False,
                temperature=0.2,
                max_new_tokens=128,
                use_cache=True,
                stopping_criteria=[stopping_criteria],
            )
        response = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        
        # Reward function: if the response indicates a strong trailer scene, positive reward
        reward = 0
        if "important" in response.lower():
            reward = 1  # Positive reward for selecting key scenes
        elif "emotional" in response.lower():
            reward = 2  # Positive reward for emotional scenes
        else:
            reward = -1  # Negative reward for scenes that don't contribute to the trailer
        
        done = False  # Keep interacting until the trailer is generated (you can adjust the stopping condition)
        return self.video, reward, done, {}

# 4. **Example Usage**:
#    - We will instantiate the `MovieTrailerEnv` class, reset the environment, and then allow the agent to select scenes and generate a trailer.

video_path = "./examples/video1.mp4"  # Make sure the video path is correct
env = MovieTrailerEnv(video_path)

# Reset the environment (starting point)
env.reset()

# Simulate the agent selecting a frame to generate a trailer scene
action = 5  # Example: the agent selects the 5th key frame
next_state, reward, done, _ = env.step(action)

# Display the generated response and reward for the selected scene
print("Generated Response:", next_state)
print("Reward:", reward)

# 5. **Explanation of the Key Components:**
#    - **State Space**: Represents the video frames, where each frame is a potential scene the agent can select.
#    - **Action Space**: The set of all possible frames (scenes) that the agent can select.
#    - **Reward Function**: Positive rewards are given for scenes that are deemed important for the trailer, while negative rewards are given for irrelevant scenes.
#    - **Prompt Generation**: Dynamic prompts are generated based on scene content (e.g., action, emotional). These prompts are then passed to LongVU for analysis.

# 6. **Next Steps:**
#    - Enhance the reward function by considering more sophisticated criteria (e.g., engagement, pacing).
#    - Implement scene segmentation to better identify key moments in the video.
#    - Allow the agent to choose multiple scenes over multiple steps to generate a full trailer.

# The agent can now iteratively select scenes for a movie trailer and receive rewards based on how well the scenes fit.
