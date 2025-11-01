import { Modality, FunctionDeclaration } from "@google/genai";

export type SupportedModels =
  | 'gemini-2.5-flash'
  | 'gemini-2.5-flash-lite'
  | 'gemini-2.5-pro'
  | 'imagen-4.0-generate-001'
  | 'veo-3.1-fast-generate-preview'
  | 'veo-3.1-generate-preview' // For reference images or video extension
  | 'gemini-2.5-flash-image'
  | 'gemini-2.5-flash-native-audio-preview-09-2025'
  | 'gemini-2.5-flash-preview-tts';

export enum Tab {
  TEXT_GENERATION = 'Text Generation',
  IMAGE_GENERATION = 'Image Generation',
  IMAGE_EDITING = 'Image Editing',
  VIDEO_GENERATION = 'Video Generation',
  VIDEO_UNDERSTANDING = 'Video Understanding',
  LIVE_CHAT = 'Live Chat',
  TEXT_TO_SPEECH = 'Text-to-Speech',
  CHATBOT = 'Chatbot',
  GROUNDING = 'Grounding'
}

export interface ChatMessage {
  role: 'user' | 'model';
  content: string;
}

export type AspectRatio = '1:1' | '3:4' | '4:3' | '9:16' | '16:9';

export type VideoAspectRatio = '16:9' | '9:16';

export type VideoResolution = '720p' | '1080p';

export interface GroundingChunk {
  web?: {
    uri: string;
    title: string;
  };
  maps?: {
    uri: string;
    title: string;
    placeAnswerSources?: {
      reviewSnippets?: {
        reviewSnippet: string;
        uri: string;
      }[];
    };
  };
}

export interface ImageInput {
  base64Data: string;
  mimeType: string;
}

export type GeminiAPICallback = (response: any) => void;

// Audio decoding utilities
export declare function decode(base64: string): Uint8Array;
export declare function decodeAudioData(
  data: Uint8Array,
  ctx: AudioContext,
  sampleRate: number,
  numChannels: number,
): Promise<AudioBuffer>;
export declare function encode(bytes: Uint8Array): string;

export interface ToolFunctionDeclaration {
  functionDeclarations: FunctionDeclaration[];
}
