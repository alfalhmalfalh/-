import { GoogleGenAI, Modality, Blob as GeminiBlob, GenerateContentResponse, GenerateContentParameters, ThinkingConfig, Tool, GenerateVideosOperation } from "@google/genai";
import { SupportedModels, AspectRatio, VideoAspectRatio, VideoResolution, GroundingChunk, ImageInput } from '../types';
import { VEO_BILLING_DOCS_LINK } from '../constants';

// Helper functions for audio encoding/decoding (from Gemini docs)
export function decode(base64: string): Uint8Array {
  const binaryString = atob(base64);
  const len = binaryString.length;
  const bytes = new Uint8Array(len);
  for (let i = 0; i < len; i++) {
    bytes[i] = binaryString.charCodeAt(i);
  }
  return bytes;
}

export async function decodeAudioData(
  data: Uint8Array,
  ctx: AudioContext,
  sampleRate: number,
  numChannels: number,
): Promise<AudioBuffer> {
  const dataInt16 = new Int16Array(data.buffer);
  const frameCount = dataInt16.length / numChannels;
  const buffer = ctx.createBuffer(numChannels, frameCount, sampleRate);

  for (let channel = 0; channel < numChannels; channel++) {
    const channelData = buffer.getChannelData(channel);
    for (let i = 0; i < frameCount; i++) {
      channelData[i] = dataInt16[i * numChannels + channel] / 32768.0;
    }
  }
  return buffer;
}

export function encode(bytes: Uint8Array): string {
  let binary = '';
  const len = bytes.byteLength;
  for (let i = 0; i < len; i++) {
    binary += String.fromCharCode(bytes[i]);
  }
  return btoa(binary);
}

export function createPcmBlob(data: Float32Array, sampleRate: number): GeminiBlob {
  const l = data.length;
  const int16 = new Int16Array(l);
  for (let i = 0; i < l; i++) {
    int16[i] = data[i] * 32768;
  }
  return {
    data: encode(new Uint8Array(int16.buffer)),
    mimeType: `audio/pcm;rate=${sampleRate}`,
  };
}

export function fileToBase64(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => {
      if (typeof reader.result === 'string') {
        // Extract base64 part only
        const base64 = reader.result.split(',')[1];
        resolve(base64);
      } else {
        reject(new Error("Failed to convert file to base64 string."));
      }
    };
    reader.onerror = (error) => reject(error);
    reader.readAsDataURL(file);
  });
}

// Function to get a new Gemini client, especially for Veo due to API key selection.
export function getGeminiClient(): GoogleGenAI {
  if (!process.env.API_KEY) {
    console.error("API_KEY is not defined.");
    throw new Error("Gemini API key is not configured.");
  }
  // Corrected: Always use a named parameter for the API key.
  return new GoogleGenAI({ apiKey: process.env.API_KEY });
}

// Veo API Key Selection Check
export async function checkAndSelectVeoApiKey(): Promise<boolean> {
  if (typeof window === 'undefined' || !window.aistudio || !window.aistudio.hasSelectedApiKey) {
    console.warn("window.aistudio not available. Assuming API key is selected or not required for this environment.");
    return true; // Assume success in non-AISTudio environments
  }

  const hasKey = await window.aistudio.hasSelectedApiKey();
  if (!hasKey) {
    console.warn("Veo API key not selected. Opening key selection dialog.");
    await window.aistudio.openSelectKey();
    // Assume success after opening the dialog for the current operation attempt,
    // as the race condition suggests it might not be immediately true.
    return true;
  }
  return true;
}

// Unified error handler for API calls
export const handleApiError = (error: any): string => {
  console.error("Gemini API error:", error);
  let errorMessage = "An unexpected error occurred.";
  if (error.message) {
    errorMessage = error.message;
  } else if (error.response && error.response.status) {
    errorMessage = `API Error: ${error.response.status} ${error.response.statusText}`;
    if (error.response.data && error.response.data.message) {
      errorMessage += ` - ${error.response.data.message}`;
    }
  }

  // Specific handling for "Requested entity was not found." for Veo key reset
  if (errorMessage.includes("Requested entity was not found.")) {
    errorMessage += " This might indicate an issue with your selected API key. Please try selecting it again.";
    if (typeof window !== 'undefined' && window.aistudio && window.aistudio.openSelectKey) {
      // Re-prompt key selection if this specific error occurs
      window.aistudio.openSelectKey().catch(e => console.error("Error opening key selection dialog:", e));
    }
  }

  return errorMessage;
};

// Function to extract grounding chunks
export function extractGroundingChunks(response: GenerateContentResponse): GroundingChunk[] {
  const groundingChunks: GroundingChunk[] = [];
  if (response.candidates?.[0]?.groundingMetadata?.groundingChunks) {
    for (const chunk of response.candidates[0].groundingMetadata.groundingChunks) {
      // Ensure uri and title exist before adding to array to match GroundingChunk type
      if (chunk.web && chunk.web.uri && chunk.web.title) {
        groundingChunks.push({ web: chunk.web });
      } else if (chunk.maps && chunk.maps.uri && chunk.maps.title) {
        // Fix: Explicitly construct the maps object to match the GroundingChunk type,
        // casting `snippet` to its expected type based on the guidelines, as the SDK's internal type
        // `GroundingChunkMapsPlaceAnswerSourcesReviewSnippet` may be more generic or different.
        const mapsData: GroundingChunk['maps'] = {
          uri: chunk.maps.uri,
          title: chunk.maps.title,
        };
        if (chunk.maps.placeAnswerSources?.reviewSnippets) {
          mapsData.placeAnswerSources = {
            reviewSnippets: chunk.maps.placeAnswerSources.reviewSnippets.map(snippet => ({
              reviewSnippet: (snippet as { reviewSnippet: string; uri: string }).reviewSnippet,
              uri: (snippet as { reviewSnippet: string; uri: string }).uri,
            })),
          };
        }
        groundingChunks.push({ maps: mapsData });
      }
    }
  }
  return groundingChunks;
}

// Generic `generateContent` call
export async function generateContent(
  model: SupportedModels,
  prompt: string,
  config?: {
    systemInstruction?: string;
    thinkingBudget?: number;
    tools?: Tool[];
    images?: ImageInput[];
    audio?: { base64Data: string; mimeType: string };
    geolocation?: GeolocationPosition | null;
  }
): Promise<GenerateContentResponse> {
  const ai = getGeminiClient();
  const parts: GenerateContentParameters['contents'] = [];

  if (config?.images && config.images.length > 0) {
    parts.push(...config.images.map(img => ({ inlineData: { data: img.base64Data, mimeType: img.mimeType } })));
  }
  if (config?.audio) {
    parts.push({ inlineData: { data: config.audio.base64Data, mimeType: config.audio.mimeType } });
  }

  if (prompt) {
    parts.push({ text: prompt });
  }

  const generateConfig: GenerateContentParameters['config'] = {
    systemInstruction: config?.systemInstruction,
    thinkingConfig: config?.thinkingBudget !== undefined ? { thinkingBudget: config.thinkingBudget } : undefined,
    tools: config?.tools,
  };

  if (config?.tools?.some(tool => (tool as any).googleMaps)) {
    if (config.geolocation) {
      generateConfig.toolConfig = {
        retrievalConfig: {
          latLng: {
            latitude: config.geolocation.coords.latitude,
            longitude: config.geolocation.coords.longitude
          }
        }
      };
    } else {
      console.warn("Geolocation not provided for Google Maps grounding.");
    }
  }

  return await ai.models.generateContent({
    model: model,
    contents: parts, // Pass the array of parts directly
    config: generateConfig,
  });
}