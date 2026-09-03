import os
import asyncio
import base64
import io
import subprocess
import tempfile
from typing import Dict, List, Optional
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from anthropic import Anthropic
from elevenlabs import ElevenLabs
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="AI Debate Arena")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize API clients
anthropic_client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
elevenlabs_client = ElevenLabs(api_key=os.getenv("ELEVENLABS_API_KEY"))

# Default voice IDs by gender
VOICE_IDS = {
    "male": os.getenv("MALE_VOICE_ID", "onwK4e9ZLuTAKqWW03F9"),  # Male voice
    "female": os.getenv("FEMALE_VOICE_ID", "EXAVITQu4vr4xnSDxMaL"),  # Female voice
}

# Personality configurations with gender-based names and voice settings
PERSONALITY_CONFIG = {
    "Logical and analytical": {
        "names": {"male": "Dr. James Watson", "female": "Dr. Sarah Chen"},
        "voice_settings": {"stability": 0.7, "similarity_boost": 0.75}
    },
    "Passionate and emotional": {
        "names": {"male": "Marcus Rivera", "female": "Sofia Russo"},
        "voice_settings": {"stability": 0.5, "similarity_boost": 0.75}
    },
    "Sarcastic and witty": {
        "names": {"male": "Danny Blake", "female": "Casey Morgan"},
        "voice_settings": {"stability": 0.3, "similarity_boost": 0.75}
    },
    "Calm and philosophical": {
        "names": {"male": "Sage Nakamura", "female": "Luna Park"},
        "voice_settings": {"stability": 0.8, "similarity_boost": 0.75}
    },
    "Aggressive and assertive": {
        "names": {"male": "Victor Kane", "female": "Diana Storm"},
        "voice_settings": {"stability": 0.3, "similarity_boost": 0.75}
    },
    "Empathetic and thoughtful": {
        "names": {"male": "David Torres", "female": "Emma Hayes"},
        "voice_settings": {"stability": 0.5, "similarity_boost": 0.75}
    },
    "Humorous and entertaining": {
        "names": {"male": "Charlie West", "female": "Riley Jones"},
        "voice_settings": {"stability": 0.3, "similarity_boost": 0.75}
    },
    "Academic and scholarly": {
        "names": {"male": "Dr. Robert Kim", "female": "Dr. Michelle Lee"},
        "voice_settings": {"stability": 0.8, "similarity_boost": 0.75}
    }
}

# Store cloned voices temporarily
cloned_voices = {}


async def generate_argument(
    topic: str,
    personality: str,
    debater_name: str,
    opponent_name: str,
    round_number: int,
    is_final: bool,
    debate_history: List[Dict],
    position: str,
    websocket
) -> tuple[str, bool, Optional[str]]:
    """Generate debate argument using Claude with web search. Returns (text, used_search, search_query)."""

    # Build context from debate history
    history_context = ""
    if debate_history:
        history_context = "\n\nConversation so far:\n"
        for entry in debate_history:
            history_context += f"{entry['name']}: {entry['argument']}\n"

    # Create system prompt - strict length control via prompt only
    system_prompt = f"""You are {debater_name}, a debater with the following personality: {personality}.
You are arguing {position} the topic: "{topic}"
Your opponent is {opponent_name}.

CRITICAL RULE - READ THIS CAREFULLY:
You MUST respond in exactly 2-3 sentences. No more. This is a fast-paced live debate - be sharp and concise. Every sentence should hit hard. If you go longer than 3 sentences, you lose the audience.

Each sentence must be punchy and under 25 words. Directly engage with {opponent_name}'s latest point.

You have access to web search. Use it when a specific recent fact, statistic, or event would strengthen your argument. Don't search every turn - only when real evidence would make a difference.

Stay in character with your {personality} personality style. Make every word count."""

    # Create user prompt - no phase labels
    if is_final:
        user_prompt = f"Wrap up your argument {position} the topic. Make your final point count." + history_context
    elif round_number == 1:
        user_prompt = f"Introduce your position {position} the topic." + history_context
    else:
        user_prompt = f"Respond to {opponent_name} and advance your argument {position} the topic." + history_context

    # Call Claude API with web search tool - no token limit to avoid cutoffs
    response = anthropic_client.messages.create(
        model="claude-sonnet-4-5-20250929",
        max_tokens=1024,  # High limit - length controlled by prompt only
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}],
        tools=[{"type": "web_search_20250305", "name": "web_search"}]
    )

    # Track if search was used
    used_search = False
    search_query = None

    # Handle tool use (web search)
    messages = [{"role": "user", "content": user_prompt}]

    while response.stop_reason == "tool_use":
        used_search = True

        # Extract tool use blocks and search query
        assistant_content = []
        for block in response.content:
            if hasattr(block, "text"):
                assistant_content.append({"type": "text", "text": block.text})
            elif block.type == "tool_use":
                # Extract search query from tool input
                if block.name == "web_search" and hasattr(block, "input"):
                    search_query = block.input.get("query", "searching for information")

                assistant_content.append({
                    "type": "tool_use",
                    "id": block.id,
                    "name": block.name,
                    "input": block.input
                })

        messages.append({"role": "assistant", "content": assistant_content})

        # Add tool results (Claude handles search automatically)
        tool_results = []
        for block in response.content:
            if block.type == "tool_use":
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": "Search completed"
                })

        messages.append({"role": "user", "content": tool_results})

        # Continue conversation
        response = anthropic_client.messages.create(
            model="claude-sonnet-4-5-20250929",
            max_tokens=1024,
            system=system_prompt,
            messages=messages,
            tools=[{"type": "web_search_20250305", "name": "web_search"}]
        )

    # Extract final text response
    text_content = ""
    for block in response.content:
        if hasattr(block, "text"):
            text_content += block.text

    return text_content, used_search, search_query


async def text_to_speech(text: str, voice_id: str, voice_settings: dict = None) -> bytes:
    """Convert text to speech using ElevenLabs with custom voice settings."""

    try:
        if voice_settings:
            # Use custom settings for default voices - pass as dict
            audio = elevenlabs_client.text_to_speech.convert(
                voice_id=voice_id,
                text=text,
                model_id="eleven_turbo_v2_5",
                voice_settings=voice_settings
            )
        else:
            # For cloned voices, use default settings
            audio = elevenlabs_client.text_to_speech.convert(
                voice_id=voice_id,
                text=text,
                model_id="eleven_turbo_v2_5"
            )
    except Exception as e:
        import traceback
        print(f"TTS error: {e}")
        print(f"TTS traceback:\n{traceback.format_exc()}")
        # Fallback - simplest possible call
        audio = elevenlabs_client.text_to_speech.convert(
            voice_id=voice_id,
            text=text
        )

    # Collect audio bytes
    audio_bytes = b""
    for chunk in audio:
        audio_bytes += chunk

    return audio_bytes


@app.get("/")
async def root():
    return {"message": "AI Debate Arena API"}


def convert_audio_to_mp3(audio_bytes: bytes) -> bytes:
    """Convert audio to MP3 using ffmpeg"""
    try:
        # Create temporary files
        with tempfile.NamedTemporaryFile(suffix='.webm', delete=False) as temp_input:
            temp_input.write(audio_bytes)
            temp_input_path = temp_input.name

        temp_output_path = tempfile.mktemp(suffix='.mp3')

        # Use ffmpeg to convert
        subprocess.run([
            'ffmpeg',
            '-i', temp_input_path,
            '-vn',  # No video
            '-ar', '44100',  # Audio sample rate
            '-ac', '2',  # Stereo
            '-b:a', '128k',  # Bitrate
            temp_output_path
        ], check=True, capture_output=True)

        # Read converted file
        with open(temp_output_path, 'rb') as f:
            mp3_bytes = f.read()

        # Cleanup
        os.unlink(temp_input_path)
        os.unlink(temp_output_path)

        return mp3_bytes
    except Exception as e:
        print(f"FFmpeg conversion failed: {e}")
        # Return original if conversion fails
        return audio_bytes


@app.post("/api/clone-voice")
async def clone_voice(
    name: str = Form(...),
    audio: UploadFile = File(...)
):
    """Clone a voice using ElevenLabs API"""
    try:
        # Read audio file
        audio_bytes = await audio.read()

        # Convert WebM to MP3 for ElevenLabs compatibility
        final_audio = convert_audio_to_mp3(audio_bytes)

        # Call ElevenLabs voice cloning API
        response = elevenlabs_client.voices.add(
            name=name,
            files=[final_audio]
        )

        voice_id = response.voice_id

        print(f"Voice cloned successfully: {name} -> {voice_id}")

        # Store in memory
        cloned_voices[voice_id] = name

        return {
            "success": True,
            "voice_id": voice_id,
            "name": name
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


@app.delete("/api/clone-voice/{voice_id}")
async def delete_cloned_voice(voice_id: str):
    """Delete a cloned voice"""
    try:
        elevenlabs_client.voices.delete(voice_id)
        if voice_id in cloned_voices:
            del cloned_voices[voice_id]
        return {"success": True}
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.websocket("/ws/debate")
async def debate_websocket(websocket: WebSocket):
    await websocket.accept()

    try:
        # Receive debate configuration
        config = await websocket.receive_json()
        topic = config["topic"]
        personality1 = config["personality1"]
        personality2 = config["personality2"]
        gender1 = config.get("gender1", "male")
        gender2 = config.get("gender2", "female")
        name1 = config.get("name1") or PERSONALITY_CONFIG[personality1]["names"][gender1]
        name2 = config.get("name2") or PERSONALITY_CONFIG[personality2]["names"][gender2]
        voice_id1 = config.get("voice_id1") or VOICE_IDS[gender1]
        voice_id2 = config.get("voice_id2") or VOICE_IDS[gender2]

        print(f"Debate config: voice_id1={voice_id1}, voice_id2={voice_id2}, names={name1}, {name2}")

        # Fixed: 3 chances per person = 6 total exchanges
        total_rounds = 3  # Each person gets 3 turns

        # Get voice settings
        voice_settings1 = PERSONALITY_CONFIG[personality1]["voice_settings"]
        voice_settings2 = PERSONALITY_CONFIG[personality2]["voice_settings"]

        debate_history = []
        paused = False

        await websocket.send_json({
            "type": "status",
            "message": "Debate starting..."
        })

        await websocket.send_json({
            "type": "debater_names",
            "name1": name1,
            "name2": name2
        })

        for round_number in range(1, total_rounds + 1):
            # Check for pause/stop commands
            try:
                # Non-blocking check for messages
                message = await asyncio.wait_for(
                    websocket.receive_json(),
                    timeout=0.1
                )
                if message.get("type") == "pause":
                    paused = True
                    await websocket.send_json({
                        "type": "paused",
                        "message": "Debate paused"
                    })
                    break
                elif message.get("type") == "stop":
                    await websocket.send_json({
                        "type": "stopped",
                        "message": "Debate stopped"
                    })
                    return
            except asyncio.TimeoutError:
                pass  # No command received, continue

            await websocket.send_json({
                "type": "round_start",
                "round_number": round_number,
                "total_rounds": total_rounds
            })

            # Debater 1 speaks
            await websocket.send_json({
                "type": "speaking",
                "debater": 1,
                "name": name1
            })

            is_final = (round_number == total_rounds)
            argument1, used_search1, search_query1 = await generate_argument(
                topic, personality1, name1, name2, round_number, is_final, debate_history, "FOR", websocket
            )

            debate_history.append({
                "debater": 1,
                "name": name1,
                "argument": argument1
            })

            await websocket.send_json({
                "type": "argument",
                "debater": 1,
                "name": name1,
                "text": argument1,
                "round": round_number,
                "used_search": used_search1,
                "search_query": search_query1 if used_search1 else None
            })

            # Generate and send audio for debater 1
            try:
                # Use voice settings only for default voices, not cloned ones
                is_cloned1 = voice_id1 in cloned_voices
                settings1 = None if is_cloned1 else voice_settings1
                audio1 = await text_to_speech(argument1, voice_id1, settings1)
                audio1_base64 = base64.b64encode(audio1).decode("utf-8")

                await websocket.send_json({
                    "type": "audio",
                    "debater": 1,
                    "audio": audio1_base64
                })

                # Calculate proper wait time based on text length (rough estimate: 150 words per minute)
                word_count = len(argument1.split())
                audio_duration = (word_count / 150) * 60  # Convert to seconds
                await asyncio.sleep(max(audio_duration + 1, 5))  # Add 1 second buffer, minimum 5 seconds
            except Exception as e:
                print(f"TTS error for debater 1: {e}")
                await asyncio.sleep(5)  # Fallback wait time

            # Debater 2 speaks
            await websocket.send_json({
                "type": "speaking",
                "debater": 2,
                "name": name2
            })

            argument2, used_search2, search_query2 = await generate_argument(
                topic, personality2, name2, name1, round_number, is_final, debate_history, "AGAINST", websocket
            )

            debate_history.append({
                "debater": 2,
                "name": name2,
                "argument": argument2
            })

            await websocket.send_json({
                "type": "argument",
                "debater": 2,
                "name": name2,
                "text": argument2,
                "round": round_number,
                "used_search": used_search2,
                "search_query": search_query2 if used_search2 else None
            })

            # Generate and send audio for debater 2
            try:
                # Use voice settings only for default voices, not cloned ones
                is_cloned2 = voice_id2 in cloned_voices
                settings2 = None if is_cloned2 else voice_settings2
                audio2 = await text_to_speech(argument2, voice_id2, settings2)
                audio2_base64 = base64.b64encode(audio2).decode("utf-8")

                await websocket.send_json({
                    "type": "audio",
                    "debater": 2,
                    "audio": audio2_base64
                })

                # Calculate proper wait time based on text length
                word_count = len(argument2.split())
                audio_duration = (word_count / 150) * 60  # Convert to seconds
                await asyncio.sleep(max(audio_duration + 1, 5))  # Add 1 second buffer, minimum 5 seconds
            except Exception as e:
                print(f"TTS error for debater 2: {e}")
                await asyncio.sleep(5)  # Fallback wait time

        # Debate complete
        if not paused:
            await websocket.send_json({
                "type": "debate_complete",
                "message": "Debate finished! Cast your vote."
            })

    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"Error in debate: {e}")
        print(f"Full traceback:\n{error_trace}")
        try:
            await websocket.send_json({
                "type": "error",
                "message": f"Error: {str(e)}"
            })
        except:
            pass  # WebSocket might be closed


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
