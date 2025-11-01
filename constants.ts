import { SupportedModels, Tab, AspectRatio, VideoAspectRatio, VideoResolution } from './types';

export const GEMINI_FLASH_MODEL: SupportedModels = 'gemini-2.5-flash';
export const GEMINI_FLASH_LITE_MODEL: SupportedModels = 'gemini-2.5-flash-lite';
export const GEMINI_PRO_MODEL: SupportedModels = 'gemini-2.5-pro';
export const IMAGEN_MODEL: SupportedModels = 'imagen-4.0-generate-001';
export const VEO_FAST_MODEL: SupportedModels = 'veo-3.1-fast-generate-preview';
export const VEO_GENERATE_MODEL: SupportedModels = 'veo-3.1-generate-preview'; // For reference images
export const GEMINI_FLASH_IMAGE_MODEL: SupportedModels = 'gemini-2.5-flash-image';
export const GEMINI_LIVE_AUDIO_MODEL: SupportedModels = 'gemini-2.5-flash-native-audio-preview-09-2025';
export const GEMINI_TTS_MODEL: SupportedModels = 'gemini-2.5-flash-preview-tts';

export const ALL_TABS: Tab[] = [
  Tab.TEXT_GENERATION,
  Tab.IMAGE_GENERATION,
  Tab.IMAGE_EDITING,
  Tab.VIDEO_GENERATION,
  Tab.VIDEO_UNDERSTANDING,
  Tab.LIVE_CHAT,
  Tab.TEXT_TO_SPEECH,
  Tab.CHATBOT,
  Tab.GROUNDING,
];

export const ASPECT_RATIOS: AspectRatio[] = ['1:1', '3:4', '4:3', '9:16', '16:9'];
export const VIDEO_ASPECT_RATIOS: VideoAspectRatio[] = ['16:9', '9:16'];
export const VIDEO_RESOLUTIONS: VideoResolution[] = ['720p', '1080p'];

export const VOICE_NAMES = [
  { value: 'Kore', label: 'Kore (Female)' },
  { value: 'Puck', label: 'Puck (Male)' },
  { value: 'Charon', label: 'Charon (Male)' },
  { value: 'Fenrir', label: 'Fenrir (Female)' },
  { value: 'Zephyr', label: 'Zephyr (Female)' },
];

export const DEFAULT_SYSTEM_INSTRUCTION_TEXT = 'You are a helpful AI assistant.';

export const VEO_BILLING_DOCS_LINK = 'https://ai.google.dev/gemini-api/docs/billing';

export const JPEG_QUALITY = 0.9;
export const FRAME_RATE = 5; // Frames per second for video understanding