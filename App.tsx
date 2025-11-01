import React, { useState, useEffect, useRef, useCallback, useReducer } from 'react';
import { GoogleGenAI, Modality, LiveServerMessage, Chat, Tool, Type, GenerateVideosOperation, Session } from '@google/genai';
import { fileToBase64, createPcmBlob, decode, decodeAudioData, getGeminiClient, checkAndSelectVeoApiKey, handleApiError, extractGroundingChunks, generateContent } from './services/geminiService';
import {
  ALL_TABS,
  GEMINI_FLASH_MODEL,
  GEMINI_FLASH_LITE_MODEL,
  GEMINI_PRO_MODEL,
  IMAGEN_MODEL,
  VEO_FAST_MODEL,
  GEMINI_FLASH_IMAGE_MODEL,
  GEMINI_LIVE_AUDIO_MODEL,
  GEMINI_TTS_MODEL,
  ASPECT_RATIOS,
  VIDEO_ASPECT_RATIOS,
  VIDEO_RESOLUTIONS,
  VOICE_NAMES,
  DEFAULT_SYSTEM_INSTRUCTION_TEXT,
  JPEG_QUALITY,
  FRAME_RATE,
} from './constants';
import { Tab, ChatMessage, SupportedModels, AspectRatio, VideoAspectRatio, VideoResolution, GroundingChunk, ImageInput } from './types';
import Tabs from './components/Tabs';
import LoadingSpinner from './components/LoadingSpinner';
import ErrorMessage from './components/ErrorMessage';
import TranscriptionDisplay from './components/TranscriptionDisplay';
import ImageUploader from './components/ImageUploader';
import VideoPlayer from './components/VideoPlayer';

interface AppState {
  activeTab: Tab;
  loading: boolean;
  error: string | null;
  language: 'en' | 'ar';
}

type AppAction =
  | { type: 'SET_ACTIVE_TAB'; payload: Tab }
  | { type: 'SET_LOADING'; payload: boolean }
  | { type: 'SET_ERROR'; payload: string | null }
  | { type: 'SET_LANGUAGE'; payload: 'en' | 'ar' };

const appReducer = (state: AppState, action: AppAction): AppState => {
  switch (action.type) {
    case 'SET_ACTIVE_TAB':
      return { ...state, activeTab: action.payload, error: null };
    case 'SET_LOADING':
      return { ...state, loading: action.payload };
    case 'SET_ERROR':
      return { ...state, error: action.payload };
    case 'SET_LANGUAGE':
      return { ...state, language: action.payload };
    default:
      return state;
  }
};

const App: React.FC = () => {
  const [state, dispatch] = useReducer(appReducer, {
    activeTab: Tab.TEXT_GENERATION,
    loading: false,
    error: null,
    language: 'en',
  });

  const [textPrompt, setTextPrompt] = useState<string>('');
  const [textResponse, setTextResponse] = useState<string>('');
  const [selectedTextModel, setSelectedTextModel] = useState<SupportedModels>(GEMINI_FLASH_MODEL);
  const [thinkingModeEnabled, setThinkingModeEnabled] = useState<boolean>(false);
  const [systemInstruction, setSystemInstruction] = useState<string>(DEFAULT_SYSTEM_INSTRUCTION_TEXT);

  const [imageGenPrompt, setImageGenPrompt] = useState<string>('');
  const [generatedImageUrl, setGeneratedImageUrl] = useState<string | null>(null);
  const [imageGenAspectRatio, setImageGenAspectRatio] = useState<AspectRatio>('1:1');

  const [imageEditPrompt, setImageEditPrompt] = useState<string>('');
  const [imageToEditFile, setImageToEditFile] = useState<File | null>(null);
  const [imageToEditInput, setImageToEditInput] = useState<ImageInput | null>(null);
  const [editedImageUrl, setEditedImageUrl] = useState<string | null>(null);

  const [videoGenPrompt, setVideoGenPrompt] = useState<string>('');
  const [videoGenImageFile, setVideoGenImageFile] = useState<File | null>(null);
  const [videoGenImageInput, setVideoGenImageInput] = useState<ImageInput | null>(null);
  const [generatedVideoUrl, setGeneratedVideoUrl] = useState<string | null>(null);
  const [videoGenAspectRatio, setVideoGenAspectRatio] = useState<VideoAspectRatio>('16:9');
  const [videoGenResolution, setVideoGenResolution] = useState<VideoResolution>('720p');

  const [videoUnderstandPrompt, setVideoUnderstandPrompt] = useState<string>('');
  const [videoUnderstandFile, setVideoUnderstandFile] = useState<File | null>(null);
  const [videoUnderstandFrame, setVideoUnderstandFrame] = useState<ImageInput | null>(null); // Extracted frame
  const [videoUnderstandResponse, setVideoUnderstandResponse] = useState<string>('');

  const [chatMessages, setChatMessages] = useState<ChatMessage[]>([]);
  const chatInstance = useRef<Chat | null>(null);
  const chatInputRef = useRef<HTMLTextAreaElement>(null);

  const [liveChatInputTranscription, setLiveChatInputTranscription] = useState<string>('');
  const [liveChatOutputTranscription, setLiveChatOutputTranscription] = useState<string>('');
  const [liveChatHistory, setLiveChatHistory] = useState<string[]>([]);
  const [isLiveChatActive, setIsLiveChatActive] = useState<boolean>(false);
  // Corrected: `live.connect` returns `Promise<Session>`, so useRef should be `Promise<Session>`
  const liveSessionPromise = useRef<Promise<Session> | null>(null);
  const nextStartTime = useRef<number>(0);
  const outputAudioSources = useRef<Set<AudioBufferSourceNode>>(new Set());
  const inputAudioContext = useRef<AudioContext | null>(null);
  const outputAudioContext = useRef<AudioContext | null>(null);
  const mediaStreamSource = useRef<MediaStreamAudioSourceNode | null>(null);
  const scriptProcessorNode = useRef<ScriptProcessorNode | null>(null);

  const [ttsInput, setTtsInput] = useState<string>('');
  const [selectedVoice, setSelectedVoice] = useState<string>(VOICE_NAMES[0].value);
  const audioRef = useRef<HTMLAudioElement>(null);

  const [groundingPrompt, setGroundingPrompt] = useState<string>('');
  const [groundingResponse, setGroundingResponse] = useState<string>('');
  const [groundingLinks, setGroundingLinks] = useState<GroundingChunk[]>([]);
  const [geolocation, setGeolocation] = useState<GeolocationPosition | null>(null);
  const [groundingTools, setGroundingTools] = useState<{ googleSearch: boolean; googleMaps: boolean }>({
    googleSearch: true,
    googleMaps: false,
  });

  // Geolocation effect for Maps Grounding
  useEffect(() => {
    if (state.activeTab === Tab.GROUNDING && groundingTools.googleMaps) {
      if (navigator.geolocation) {
        navigator.geolocation.getCurrentPosition(
          (position) => {
            setGeolocation(position);
            dispatch({ type: 'SET_ERROR', payload: null });
          },
          (error) => {
            dispatch({ type: 'SET_ERROR', payload: `Geolocation error: ${error.message}` });
            setGeolocation(null);
          },
          { enableHighAccuracy: true, timeout: 5000, maximumAge: 0 }
        );
      } else {
        dispatch({ type: 'SET_ERROR', payload: 'Geolocation is not supported by your browser.' });
      }
    }
  }, [state.activeTab, groundingTools.googleMaps, dispatch]);

  const resetState = useCallback(() => {
    setTextPrompt('');
    setTextResponse('');
    setImageGenPrompt('');
    setGeneratedImageUrl(null);
    setImageEditPrompt('');
    setImageToEditFile(null);
    setImageToEditInput(null);
    setEditedImageUrl(null);
    setVideoGenPrompt('');
    setVideoGenImageFile(null);
    setVideoGenImageInput(null);
    setGeneratedVideoUrl(null);
    setVideoUnderstandPrompt('');
    setVideoUnderstandFile(null);
    setVideoUnderstandFrame(null);
    setVideoUnderstandResponse('');
    setChatMessages([]);
    chatInstance.current = null;
    setLiveChatInputTranscription('');
    setLiveChatOutputTranscription('');
    setLiveChatHistory([]);
    setIsLiveChatActive(false);
    if (liveSessionPromise.current) {
      // Corrected: Access 'close' method on the resolved Session object, not the Promise itself.
      liveSessionPromise.current.then((session: Session) => session.close()).catch(console.error);
      liveSessionPromise.current = null;
    }
    if (inputAudioContext.current) inputAudioContext.current.close();
    if (outputAudioContext.current) outputAudioContext.current.close();
    inputAudioContext.current = null;
    outputAudioContext.current = null;
    mediaStreamSource.current = null;
    if (scriptProcessorNode.current) scriptProcessorNode.current.disconnect();
    scriptProcessorNode.current = null;
    nextStartTime.current = 0;
    outputAudioSources.current.forEach(source => source.stop());
    outputAudioSources.current.clear();
    setTtsInput('');
    setGroundingPrompt('');
    setGroundingResponse('');
    setGroundingLinks([]);
    setGeolocation(null);
    dispatch({ type: 'SET_ERROR', payload: null });
    dispatch({ type: 'SET_LOADING', payload: false });
  }, [dispatch]);

  useEffect(() => {
    resetState();
  }, [state.activeTab, resetState]);


  // Text Generation
  const handleTextGeneration = useCallback(async () => {
    dispatch({ type: 'SET_LOADING', payload: true });
    dispatch({ type: 'SET_ERROR', payload: null });
    setTextResponse('');

    // promptParts is not directly used in generateContent helper as it constructs parts internally
    // const promptParts = [
    //   { text: textPrompt },
    // ];

    const config: Parameters<typeof generateContent>[2] = {
      systemInstruction: systemInstruction,
    };

    if (selectedTextModel === GEMINI_PRO_MODEL && thinkingModeEnabled) {
      config.thinkingBudget = 32768; // Max for Pro
    }

    try {
      const response = await generateContent(selectedTextModel, textPrompt, config);
      setTextResponse(response.text);
    } catch (error) {
      dispatch({ type: 'SET_ERROR', payload: handleApiError(error) });
    } finally {
      dispatch({ type: 'SET_LOADING', payload: false });
    }
  }, [textPrompt, selectedTextModel, thinkingModeEnabled, systemInstruction, dispatch]);

  // Image Generation
  const handleImageGeneration = useCallback(async () => {
    dispatch({ type: 'SET_LOADING', payload: true });
    dispatch({ type: 'SET_ERROR', payload: null });
    setGeneratedImageUrl(null);

    try {
      const ai = getGeminiClient();
      const response = await ai.models.generateImages({
        model: IMAGEN_MODEL,
        prompt: imageGenPrompt,
        config: {
          numberOfImages: 1,
          outputMimeType: 'image/jpeg',
          aspectRatio: imageGenAspectRatio,
        },
      });

      const base64ImageBytes: string = response.generatedImages[0].image.imageBytes;
      setGeneratedImageUrl(`data:image/jpeg;base64,${base64ImageBytes}`);
    } catch (error) {
      dispatch({ type: 'SET_ERROR', payload: handleApiError(error) });
    } finally {
      dispatch({ type: 'SET_LOADING', payload: false });
    }
  }, [imageGenPrompt, imageGenAspectRatio, dispatch]);

  // Image Editing
  const handleImageEditing = useCallback(async () => {
    dispatch({ type: 'SET_LOADING', payload: true });
    dispatch({ type: 'SET_ERROR', payload: null });
    setEditedImageUrl(null);

    if (!imageToEditInput) {
      dispatch({ type: 'SET_ERROR', payload: 'Please upload an image to edit.' });
      dispatch({ type: 'SET_LOADING', payload: false });
      return;
    }

    try {
      const ai = getGeminiClient();
      const response = await ai.models.generateContent({
        model: GEMINI_FLASH_IMAGE_MODEL,
        contents: [ // Changed to array of Content objects for consistency
          {
            parts: [
              {
                inlineData: {
                  data: imageToEditInput.base64Data,
                  mimeType: imageToEditInput.mimeType,
                },
              },
              {
                text: imageEditPrompt,
              },
            ],
          },
        ],
        config: {
          responseModalities: [Modality.IMAGE],
        },
      });

      const base64ImageBytes: string | undefined = response.candidates?.[0]?.content?.parts?.[0]?.inlineData?.data;
      if (base64ImageBytes) {
        setEditedImageUrl(`data:image/jpeg;base64,${base64ImageBytes}`);
      } else {
        dispatch({ type: 'SET_ERROR', payload: 'No edited image returned.' });
      }
    } catch (error) {
      dispatch({ type: 'SET_ERROR', payload: handleApiError(error) });
    } finally {
      dispatch({ type: 'SET_LOADING', payload: false });
    }
  }, [imageEditPrompt, imageToEditInput, dispatch]);


  // Video Generation
  const handleVideoGeneration = useCallback(async () => {
    dispatch({ type: 'SET_LOADING', payload: true });
    dispatch({ type: 'SET_ERROR', payload: null });
    setGeneratedVideoUrl(null);

    const isKeySelected = await checkAndSelectVeoApiKey();
    if (!isKeySelected) {
      dispatch({ type: 'SET_ERROR', payload: 'API Key not selected. Please select your API key with billing enabled.' });
      dispatch({ type: 'SET_LOADING', payload: false });
      return;
    }

    try {
      const ai = getGeminiClient(); // Re-instantiate for updated API key
      let operation;

      if (videoGenImageInput) {
        // Generate video from image + prompt
        operation = await ai.models.generateVideos({
          model: VEO_FAST_MODEL,
          prompt: videoGenPrompt,
          image: {
            imageBytes: videoGenImageInput.base64Data,
            mimeType: videoGenImageInput.mimeType,
          },
          config: {
            numberOfVideos: 1,
            resolution: videoGenResolution,
            aspectRatio: videoGenAspectRatio,
          },
        });
      } else {
        // Generate video from prompt only
        operation = await ai.models.generateVideos({
          model: VEO_FAST_MODEL,
          prompt: videoGenPrompt,
          config: {
            numberOfVideos: 1,
            resolution: videoGenResolution,
            aspectRatio: videoGenAspectRatio,
          },
        });
      }


      let currentOperation: GenerateVideosOperation = operation;
      dispatch({ type: 'SET_LOADING', payload: true });
      dispatch({ type: 'SET_ERROR', payload: 'Generating video... This may take a few minutes.' });

      while (!currentOperation.done) {
        await new Promise(resolve => setTimeout(resolve, 10000)); // Poll every 10 seconds
        currentOperation = await ai.operations.getVideosOperation({ operation: currentOperation });
        if (currentOperation.error) {
          throw new Error(currentOperation.error.message || 'Video generation failed.');
        }
      }

      const downloadLink = currentOperation.response?.generatedVideos?.[0]?.video?.uri;
      if (downloadLink) {
        // Fix: Use process.env.API_KEY directly as it's the source for initialization.
        // `aiWithKey.apiKey` is private.
        const videoResponse = await fetch(`${downloadLink}&key=${process.env.API_KEY}`);
        if (!videoResponse.ok) {
          throw new Error(`Failed to fetch video: ${videoResponse.statusText}`);
        }
        const videoBlob = await videoResponse.blob();
        setGeneratedVideoUrl(URL.createObjectURL(videoBlob));
        dispatch({ type: 'SET_ERROR', payload: null });
      } else {
        dispatch({ type: 'SET_ERROR', payload: 'No video URI returned.' });
      }
    } catch (error) {
      dispatch({ type: 'SET_ERROR', payload: handleApiError(error) });
    } finally {
      dispatch({ type: 'SET_LOADING', payload: false });
    }
  }, [videoGenPrompt, videoGenImageInput, videoGenAspectRatio, videoGenResolution, dispatch]);

  // Video Understanding
  const handleVideoUnderstanding = useCallback(async () => {
    dispatch({ type: 'SET_LOADING', payload: true });
    dispatch({ type: 'SET_ERROR', payload: null });
    setVideoUnderstandResponse('');

    if (!videoUnderstandFrame) {
      dispatch({ type: 'SET_ERROR', payload: 'Please upload a video to extract a frame for understanding.' });
      dispatch({ type: 'SET_LOADING', payload: false });
      return;
    }

    try {
      const response = await generateContent(
        GEMINI_PRO_MODEL,
        videoUnderstandPrompt,
        {
          images: [videoUnderstandFrame],
          systemInstruction: 'You are an expert video analyst. Analyze the provided image frame and context.',
        }
      );
      setVideoUnderstandResponse(response.text);
    } catch (error) {
      dispatch({ type: 'SET_ERROR', payload: handleApiError(error) });
    } finally {
      dispatch({ type: 'SET_LOADING', payload: false });
    }
  }, [videoUnderstandPrompt, videoUnderstandFrame, dispatch]);


  // Chatbot
  const initializeChat = useCallback(async () => {
    if (chatInstance.current) return;
    dispatch({ type: 'SET_LOADING', payload: true });
    dispatch({ type: 'SET_ERROR', payload: null });
    try {
      const ai = getGeminiClient();
      chatInstance.current = ai.chats.create({
        model: GEMINI_FLASH_MODEL,
        config: {
          systemInstruction: 'You are a friendly and helpful chatbot assistant.',
        },
      });
      console.log("Chat initialized.");
    } catch (error) {
      dispatch({ type: 'SET_ERROR', payload: handleApiError(error) });
    } finally {
      dispatch({ type: 'SET_LOADING', payload: false });
    }
  }, [dispatch]);

  const handleChatMessageSend = useCallback(async () => {
    if (!chatInputRef.current?.value.trim()) return;
    const userMessage = chatInputRef.current.value.trim();
    setChatMessages((prev) => [...prev, { role: 'user', content: userMessage }]);
    chatInputRef.current.value = '';

    if (!chatInstance.current) {
      await initializeChat();
      if (!chatInstance.current) { // If initialization failed
        setChatMessages((prev) => [...prev.slice(0, -1)]); // Remove user message if chat failed to initialize
        return;
      }
    }

    dispatch({ type: 'SET_LOADING', payload: true });
    dispatch({ type: 'SET_ERROR', payload: null });

    try {
      const responseStream = await chatInstance.current.sendMessageStream({ message: userMessage });
      let fullResponse = '';
      for await (const chunk of responseStream) {
        fullResponse += chunk.text;
        setChatMessages((prev) => {
          const lastMessage = prev[prev.length - 1];
          if (lastMessage && lastMessage.role === 'model') {
            return [...prev.slice(0, -1), { ...lastMessage, content: fullResponse }];
          } else {
            return [...prev, { role: 'model', content: fullResponse }];
          }
        });
      }
    } catch (error) {
      dispatch({ type: 'SET_ERROR', payload: handleApiError(error) });
      setChatMessages((prev) => [...prev.slice(0, -1)]); // Remove user message if error
    } finally {
      dispatch({ type: 'SET_LOADING', payload: false });
    }
  }, [initializeChat, dispatch]);


  // Live Chat (Gemini 2.5 Native Audio)
  const setupLiveChat = useCallback(async () => {
    dispatch({ type: 'SET_LOADING', payload: true });
    dispatch({ type: 'SET_ERROR', payload: null });
    setLiveChatInputTranscription('');
    setLiveChatOutputTranscription('');
    setLiveChatHistory([]);
    nextStartTime.current = 0;

    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      dispatch({ type: 'SET_ERROR', payload: 'Microphone access is not supported in this browser.' });
      dispatch({ type: 'SET_LOADING', payload: false });
      return;
    }

    try {
      const ai = getGeminiClient();
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      // Corrected: Use standard AudioContext instead of window.webkitAudioContext
      inputAudioContext.current = new AudioContext({ sampleRate: 16000 });
      // Corrected: Use standard AudioContext instead of window.webkitAudioContext
      outputAudioContext.current = new AudioContext({ sampleRate: 24000 });
      const inputNode = inputAudioContext.current.createGain(); // Not directly used, but good practice
      const outputNode = outputAudioContext.current.createGain();
      outputNode.connect(outputAudioContext.current.destination);


      liveSessionPromise.current = ai.live.connect({
        model: GEMINI_LIVE_AUDIO_MODEL,
        callbacks: {
          onopen: () => {
            console.log('Live session opened.');
            setIsLiveChatActive(true);
            dispatch({ type: 'SET_LOADING', payload: false });

            // Stream audio from the microphone to the model.
            mediaStreamSource.current = inputAudioContext.current!.createMediaStreamSource(stream);
            scriptProcessorNode.current = inputAudioContext.current!.createScriptProcessor(4096, 1, 1);
            scriptProcessorNode.current.onaudioprocess = (audioProcessingEvent) => {
              const inputData = audioProcessingEvent.inputBuffer.getChannelData(0);
              const pcmBlob = createPcmBlob(inputData, inputAudioContext.current!.sampleRate);
              // CRITICAL: Solely rely on sessionPromise resolves and explicitly type session
              liveSessionPromise.current!.then((session: Session) => { 
                session.sendRealtimeInput({ media: pcmBlob });
              });
            };
            mediaStreamSource.current.connect(scriptProcessorNode.current);
            scriptProcessorNode.current.connect(inputAudioContext.current!.destination);
          },
          onmessage: async (message: LiveServerMessage) => {
            // Transcription handling
            if (message.serverContent?.outputTranscription) {
              setLiveChatOutputTranscription((prev) => prev + message.serverContent!.outputTranscription!.text);
            }
            if (message.serverContent?.inputTranscription) {
              setLiveChatInputTranscription((prev) => prev + message.serverContent!.inputTranscription!.text);
            }
            // IMPORTANT: Capture current transcription state before clearing
            if (message.serverContent?.turnComplete) {
              const currentInput = liveChatInputTranscription;
              const currentOutput = liveChatOutputTranscription;
              // Fix: Explicitly cast currentInput and currentOutput to string to satisfy the compiler.
              // This addresses a potential type inference issue within the asynchronous callback.
              setLiveChatHistory((prev) => [
                ...prev,
                currentInput as string, 
                currentOutput as string,
              ]);
              setLiveChatInputTranscription('');
              setLiveChatOutputTranscription('');
            }

            // Audio output handling
            const base64EncodedAudioString = message.serverContent?.modelTurn?.parts?.[0]?.inlineData?.data;
            if (base64EncodedAudioString && outputAudioContext.current) {
              nextStartTime.current = Math.max(nextStartTime.current, outputAudioContext.current.currentTime);
              const audioBuffer = await decodeAudioData(
                decode(base64EncodedAudioString),
                outputAudioContext.current,
                24000,
                1,
              );
              const source = outputAudioContext.current.createBufferSource();
              source.buffer = audioBuffer;
              source.connect(outputNode);
              source.addEventListener('ended', () => {
                outputAudioSources.current.delete(source);
              });

              source.start(nextStartTime.current);
              nextStartTime.current = nextStartTime.current + audioBuffer.duration;
              outputAudioSources.current.add(source);
            }

            // Interruption handling
            const interrupted = message.serverContent?.interrupted;
            if (interrupted) {
              for (const source of outputAudioSources.current.values()) {
                source.stop();
                outputAudioSources.current.delete(source);
              }
              nextStartTime.current = 0;
            }
          },
          onerror: (e: Event) => {
            const errorEvent = e as ErrorEvent;
            console.error('Live session error:', errorEvent.message);
            dispatch({ type: 'SET_ERROR', payload: `Live session error: ${errorEvent.message}` });
            setIsLiveChatActive(false);
            dispatch({ type: 'SET_LOADING', payload: false });
          },
          onclose: (e: CloseEvent) => {
            console.log('Live session closed:', e.code, e.reason);
            setIsLiveChatActive(false);
            dispatch({ type: 'SET_LOADING', payload: false });
            // Clean up audio resources
            if (mediaStreamSource.current) mediaStreamSource.current.disconnect();
            if (scriptProcessorNode.current) scriptProcessorNode.current.disconnect();
            if (inputAudioContext.current) inputAudioContext.current.close();
            if (outputAudioContext.current) outputAudioContext.current.close();
            inputAudioContext.current = null;
            outputAudioContext.current = null;
            mediaStreamSource.current = null;
            scriptProcessorNode.current = null;
            outputAudioSources.current.forEach(source => source.stop());
            outputAudioSources.current.clear();
          },
        },
        config: {
          responseModalities: [Modality.AUDIO],
          speechConfig: {
            voiceConfig: { prebuiltVoiceConfig: { voiceName: selectedVoice } },
          },
          systemInstruction: 'You are a helpful and friendly AI assistant, ready to chat.',
          inputAudioTranscription: {}, // Enable transcription for user input audio.
          outputAudioTranscription: {}, // Enable transcription for model output audio.
        },
      });
    } catch (err) {
      const error = err as Error;
      console.error('Failed to set up live chat:', error);
      dispatch({ type: 'SET_ERROR', payload: `Failed to access microphone or set up live chat: ${error.message}` });
      dispatch({ type: 'SET_LOADING', payload: false });
      setIsLiveChatActive(false);
    }
  }, [dispatch, liveChatInputTranscription, liveChatOutputTranscription, selectedVoice]);

  const stopLiveChat = useCallback(() => {
    if (liveSessionPromise.current) {
      // Corrected: Access 'close' method on the resolved Session object, not the Promise itself.
      liveSessionPromise.current.then((session: Session) => session.close()).catch(console.error);
      liveSessionPromise.current = null;
    }
    setIsLiveChatActive(false);
    dispatch({ type: 'SET_LOADING', payload: false });
  }, [dispatch]);


  // Text-to-Speech
  const handleTextToSpeech = useCallback(async () => {
    dispatch({ type: 'SET_LOADING', payload: true });
    dispatch({ type: 'SET_ERROR', payload: null });

    if (!ttsInput.trim()) {
      dispatch({ type: 'SET_ERROR', payload: 'Please enter text for speech generation.' });
      dispatch({ type: 'SET_LOADING', payload: false });
      return;
    }

    try {
      const ai = getGeminiClient();
      const response = await ai.models.generateContent({
        model: GEMINI_TTS_MODEL,
        contents: [{ parts: [{ text: ttsInput }] }],
        config: {
          responseModalities: [Modality.AUDIO],
          speechConfig: {
            voiceConfig: {
              prebuiltVoiceConfig: { voiceName: selectedVoice },
            },
          },
        },
      });

      const base64Audio = response.candidates?.[0]?.content?.parts?.[0]?.inlineData?.data;
      if (base64Audio) {
        // The audio bytes returned by the API is raw PCM data. It is not a standard file format like `.wav` `.mpeg`, or `.mp3`, it contains no header information.
        // It should be decoded using AudioContext.
        if (outputAudioContext.current === null) {
          outputAudioContext.current = new AudioContext({ sampleRate: 24000 });
        }
        const audioBuffer = await decodeAudioData(
          decode(base64Audio),
          outputAudioContext.current,
          24000,
          1,
        );
        const source = outputAudioContext.current.createBufferSource();
        source.buffer = audioBuffer;
        source.connect(outputAudioContext.current.destination); // Connect to default destination (speakers)
        source.start();
        
        // This part is for `audioRef` which expects a file URL, but Gemini TTS returns raw PCM.
        // Keeping it commented out or removed unless a proper way to stream raw PCM to an <audio> tag is implemented.
        // const audioBlob = new Blob([decode(base64Audio)], { type: 'audio/pcm' });
        // const url = URL.createObjectURL(audioBlob);
        // if (audioRef.current) {
        //   audioRef.current.src = url;
        //   audioRef.current.play().catch(e => console.error("Audio playback error:", e));
        // }
      } else {
        dispatch({ type: 'SET_ERROR', payload: 'No audio data received.' });
      }
    } catch (error) {
      dispatch({ type: 'SET_ERROR', payload: handleApiError(error) });
    } finally {
      dispatch({ type: 'SET_LOADING', payload: false });
    }
  }, [ttsInput, selectedVoice, dispatch]);

  // Grounding
  const handleGroundingSearch = useCallback(async () => {
    dispatch({ type: 'SET_LOADING', payload: true });
    dispatch({ type: 'SET_ERROR', payload: null });
    setGroundingResponse('');
    setGroundingLinks([]);

    const tools: Tool[] = [];
    if (groundingTools.googleSearch) {
      tools.push({ googleSearch: {} });
    }
    if (groundingTools.googleMaps) {
      tools.push({ googleMaps: {} });
      if (!geolocation) {
        dispatch({ type: 'SET_ERROR', payload: 'Geolocation is required for Google Maps grounding.' });
        dispatch({ type: 'SET_LOADING', payload: false });
        return;
      }
    }

    if (tools.length === 0) {
      dispatch({ type: 'SET_ERROR', payload: 'Please select at least one grounding tool.' });
      dispatch({ type: 'SET_LOADING', payload: false });
      return;
    }

    try {
      const response = await generateContent(
        GEMINI_FLASH_MODEL,
        groundingPrompt,
        {
          tools: tools,
          geolocation: geolocation,
        }
      );
      setGroundingResponse(response.text);
      setGroundingLinks(extractGroundingChunks(response));
    } catch (error) {
      dispatch({ type: 'SET_ERROR', payload: handleApiError(error) });
    } finally {
      dispatch({ type: 'SET_LOADING', payload: false });
    }
  }, [groundingPrompt, groundingTools, geolocation, dispatch]);


  // Common UI elements & handlers
  const renderLanguageToggle = (
    <div className="absolute top-4 right-4 z-10 flex items-center space-x-2">
      <span className="text-gray-700">Language:</span>
      <button
        onClick={() => dispatch({ type: 'SET_LANGUAGE', payload: 'en' })}
        className={`px-3 py-1 rounded-md text-sm font-medium ${
          state.language === 'en' ? 'bg-blue-500 text-white' : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
        }`}
      >
        English
      </button>
      <button
        onClick={() => dispatch({ type: 'SET_LANGUAGE', payload: 'ar' })}
        className={`px-3 py-1 rounded-md text-sm font-medium ${
          state.language === 'ar' ? 'bg-blue-500 text-white' : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
        }`}
      >
        العربية
      </button>
    </div>
  );

  return (
    <div className={`min-h-screen bg-gray-100 p-8 ${state.language === 'ar' ? 'font-arabic text-right' : 'font-sans text-left'}`} dir={state.language === 'ar' ? 'rtl' : 'ltr'}>
      {renderLanguageToggle}
      <h1 className="text-4xl font-extrabold text-center text-gray-800 mb-8 mt-4">
        Gemini Multi-Tool App
      </h1>

      <div className="max-w-6xl mx-auto bg-white shadow-xl rounded-lg overflow-hidden">
        <Tabs activeTab={state.activeTab} onTabChange={(tab) => dispatch({ type: 'SET_ACTIVE_TAB', payload: tab })} tabs={ALL_TABS}>
          {state.error && <ErrorMessage message={state.error} className="mb-4" />}
          {state.loading && <LoadingSpinner message={state.language === 'ar' ? 'جارٍ التحميل...' : 'Loading...'} />}

          {/* Text Generation Tab */}
          {state.activeTab === Tab.TEXT_GENERATION && (
            <div className="space-y-6">
              <h2 className="text-2xl font-bold text-gray-800 mb-4">{state.language === 'ar' ? 'توليد النصوص' : 'Text Generation'}</h2>

              <div>
                <label htmlFor="textModelSelect" className="block text-sm font-medium text-gray-700 mb-1">
                  {state.language === 'ar' ? 'اختر النموذج:' : 'Select Model:'}
                </label>
                <select
                  id="textModelSelect"
                  className="mt-1 block w-full pl-3 pr-10 py-2 text-base border-gray-300 focus:outline-none focus:ring-blue-500 focus:border-blue-500 sm:text-sm rounded-md"
                  value={selectedTextModel}
                  onChange={(e) => setSelectedTextModel(e.target.value as SupportedModels)}
                >
                  <option value={GEMINI_FLASH_MODEL}>{state.language === 'ar' ? 'جيميني 2.5 فلاش (سريع)' : 'Gemini 2.5 Flash (Fast)'}</option>
                  <option value={GEMINI_FLASH_LITE_MODEL}>{state.language === 'ar' ? 'جيميني 2.5 فلاش لايت (أقل زمن استجابة)' : 'Gemini 2.5 Flash Lite (Low-Latency)'}</option>
                  <option value={GEMINI_PRO_MODEL}>{state.language === 'ar' ? 'جيميني 2.5 برو (للمهام المعقدة)' : 'Gemini 2.5 Pro (Complex Tasks)'}</option>
                </select>
              </div>

              {selectedTextModel === GEMINI_PRO_MODEL && (
                <div className="flex items-center">
                  <input
                    id="thinkingMode"
                    type="checkbox"
                    className="h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 rounded"
                    checked={thinkingModeEnabled}
                    onChange={(e) => setThinkingModeEnabled(e.target.checked)}
                  />
                  <label htmlFor="thinkingMode" className={`ml-2 block text-sm font-medium text-gray-700 ${state.language === 'ar' ? 'mr-2' : ''}`}>
                    {state.language === 'ar' ? 'تفعيل وضع التفكير (أقصى ميزانية 32768 رمزاً)' : 'Enable Thinking Mode (Max budget 32768 tokens)'}
                  </label>
                </div>
              )}

              <div>
                <label htmlFor="systemInstruction" className="block text-sm font-medium text-gray-700 mb-1">
                  {state.language === 'ar' ? 'تعليمات النظام (اختياري):' : 'System Instruction (Optional):'}
                </label>
                <textarea
                  id="systemInstruction"
                  className="w-full p-3 border border-gray-300 rounded-md focus:ring-blue-500 focus:border-blue-500 resize-y"
                  rows={2}
                  value={systemInstruction}
                  onChange={(e) => setSystemInstruction(e.target.value)}
                  placeholder={state.language === 'ar' ? 'مثال: أنت مساعد إبداعي...' : 'e.g., You are a creative assistant...'}
                ></textarea>
              </div>

              <div>
                <label htmlFor="textPrompt" className="block text-sm font-medium text-gray-700 mb-1">
                  {state.language === 'ar' ? 'أدخل النص:' : 'Enter your text prompt:'}
                </label>
                <textarea
                  id="textPrompt"
                  className="w-full p-3 border border-gray-300 rounded-md focus:ring-blue-500 focus:border-blue-500 resize-y"
                  rows={6}
                  value={textPrompt}
                  onChange={(e) => setTextPrompt(e.target.value)}
                  placeholder={state.language === 'ar' ? 'أدخل استفسارك هنا، على سبيل المثال: "انشي موقع ومدونه ومتجر مع ربط بالدروبينشبنق مع لوحات الدفع ويعمل بالذكاء الصناعي كامل توليد مقالات ونشر وتوليد منتجات رقميه ونشر وتوليد وربط منتجات دروبشبينق ونشر وتواصل اجتماعي وتسويق وادارة محتوي وتحكم واحصائيات ويفعل نظام ربح اعلانات عمولات قوقل ادسنس وغيرها وقوي وفخم ودرشه ذكاء صناعي ويعمل عربي وانقليزي وقوي وللينافس وعالمي"' : 'Enter your prompt here, e.g., "Write a short story about a futuristic city."' }
                ></textarea>
              </div>

              <button
                onClick={handleTextGeneration}
                className="w-full bg-blue-600 text-white py-3 px-6 rounded-md text-lg font-semibold hover:bg-blue-700 transition-colors duration-200 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2"
                disabled={state.loading}
              >
                {state.language === 'ar' ? 'توليد النص' : 'Generate Text'}
              </button>

              {textResponse && (
                <div className="mt-6 bg-gray-50 p-4 rounded-md shadow-inner">
                  <h3 className="text-xl font-semibold text-gray-800 mb-2">{state.language === 'ar' ? 'الاستجابة:' : 'Response:'}</h3>
                  <div className="whitespace-pre-wrap text-gray-700">{textResponse}</div>
                </div>
              )}
            </div>
          )}

          {/* Image Generation Tab */}
          {state.activeTab === Tab.IMAGE_GENERATION && (
            <div className="space-y-6">
              <h2 className="text-2xl font-bold text-gray-800 mb-4">{state.language === 'ar' ? 'توليد الصور' : 'Image Generation'}</h2>
              <div>
                <label htmlFor="imageGenPrompt" className="block text-sm font-medium text-gray-700 mb-1">
                  {state.language === 'ar' ? 'أدخل وصف الصورة:' : 'Enter image description:'}
                </label>
                <textarea
                  id="imageGenPrompt"
                  className="w-full p-3 border border-gray-300 rounded-md focus:ring-blue-500 focus:border-blue-500 resize-y"
                  rows={4}
                  value={imageGenPrompt}
                  onChange={(e) => setImageGenPrompt(e.target.value)}
                  placeholder={state.language === 'ar' ? 'مثال: قطة رائد فضاء تطفو في الفضاء ببدلة ذهبية...' : 'e.g., An astronaut cat floating in space with a golden suit...'}
                ></textarea>
              </div>

              <div>
                <label htmlFor="aspectRatioSelect" className="block text-sm font-medium text-gray-700 mb-1">
                  {state.language === 'ar' ? 'نسبة العرض إلى الارتفاع:' : 'Aspect Ratio:'}
                </label>
                <select
                  id="aspectRatioSelect"
                  className="mt-1 block w-full pl-3 pr-10 py-2 text-base border-gray-300 focus:outline-none focus:ring-blue-500 focus:border-blue-500 sm:text-sm rounded-md"
                  value={imageGenAspectRatio}
                  onChange={(e) => setImageGenAspectRatio(e.target.value as AspectRatio)}
                >
                  {ASPECT_RATIOS.map((ratio) => (
                    <option key={ratio} value={ratio}>{ratio}</option>
                  ))}
                </select>
              </div>

              <button
                onClick={handleImageGeneration}
                className="w-full bg-blue-600 text-white py-3 px-6 rounded-md text-lg font-semibold hover:bg-blue-700 transition-colors duration-200 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2"
                disabled={state.loading}
              >
                {state.language === 'ar' ? 'توليد الصورة' : 'Generate Image'}
              </button>

              {generatedImageUrl && (
                <div className="mt-6 text-center">
                  <h3 className="text-xl font-semibold text-gray-800 mb-2">{state.language === 'ar' ? 'الصورة المولدة:' : 'Generated Image:'}</h3>
                  <img src={generatedImageUrl} alt="Generated" className="max-w-full h-auto mx-auto rounded-lg shadow-md border border-gray-200" />
                </div>
              )}
            </div>
          )}

          {/* Image Editing Tab */}
          {state.activeTab === Tab.IMAGE_EDITING && (
            <div className="space-y-6">
              <h2 className="text-2xl font-bold text-gray-800 mb-4">{state.language === 'ar' ? 'تحرير الصور' : 'Image Editing'}</h2>

              <ImageUploader
                label={state.language === 'ar' ? 'قم بتحميل الصورة للتحرير:' : 'Upload Image to Edit:'}
                allowedFileTypes={['image/jpeg', 'image/png', 'image/webp']}
                onFileChange={(file, imageInput) => {
                  setImageToEditFile(file);
                  setImageToEditInput(imageInput);
                }}
              />

              <div>
                <label htmlFor="imageEditPrompt" className="block text-sm font-medium text-gray-700 mb-1">
                  {state.language === 'ar' ? 'أدخل وصف التحرير:' : 'Enter edit prompt (e.g., "Add a retro filter" or "Remove the person in the background"): '}
                </label>
                <textarea
                  id="imageEditPrompt"
                  className="w-full p-3 border border-gray-300 rounded-md focus:ring-blue-500 focus:border-blue-500 resize-y"
                  rows={4}
                  value={imageEditPrompt}
                  onChange={(e) => setImageEditPrompt(e.target.value)}
                  placeholder={state.language === 'ar' ? 'مثال: أضف فلترًا عتيقًا، أو قم بإزالة الشخص في الخلفية.' : 'e.g., Add a retro filter, or Remove the person in the background.'}
                ></textarea>
              </div>

              <button
                onClick={handleImageEditing}
                className="w-full bg-blue-600 text-white py-3 px-6 rounded-md text-lg font-semibold hover:bg-blue-700 transition-colors duration-200 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2"
                disabled={state.loading || !imageToEditInput}
              >
                {state.language === 'ar' ? 'تحرير الصورة' : 'Edit Image'}
              </button>

              {editedImageUrl && (
                <div className="mt-6 text-center">
                  <h3 className="text-xl font-semibold text-gray-800 mb-2">{state.language === 'ar' ? 'الصورة المحررة:' : 'Edited Image:'}</h3>
                  <img src={editedImageUrl} alt="Edited" className="max-w-full h-auto mx-auto rounded-lg shadow-md border border-gray-200" />
                </div>
              )}
            </div>
          )}

          {/* Video Generation Tab */}
          {state.activeTab === Tab.VIDEO_GENERATION && (
            <div className="space-y-6">
              <h2 className="text-2xl font-bold text-gray-800 mb-4">{state.language === 'ar' ? 'توليد الفيديو' : 'Video Generation'}</h2>

              <ImageUploader
                label={state.language === 'ar' ? 'صورة بدء اختيارية (سيتم توليد فيديو منها):' : 'Optional starting image (will generate video from this):'}
                allowedFileTypes={['image/jpeg', 'image/png', 'image/webp']}
                onFileChange={(file, imageInput) => {
                  setVideoGenImageFile(file);
                  setVideoGenImageInput(imageInput);
                }}
              />

              <div>
                <label htmlFor="videoGenPrompt" className="block text-sm font-medium text-gray-700 mb-1">
                  {state.language === 'ar' ? 'أدخل وصف الفيديو:' : 'Enter video description:'}
                </label>
                <textarea
                  id="videoGenPrompt"
                  className="w-full p-3 border border-gray-300 rounded-md focus:ring-blue-500 focus:border-blue-500 resize-y"
                  rows={4}
                  value={videoGenPrompt}
                  onChange={(e) => setVideoGenPrompt(e.target.value)}
                  placeholder={state.language === 'ar' ? 'مثال: مدينة مستقبلية تتسابق فيها السيارات الطائرة.' : 'e.g., A futuristic city with flying cars racing.'}
                ></textarea>
              </div>

              <div>
                <label htmlFor="videoAspectRatioSelect" className="block text-sm font-medium text-gray-700 mb-1">
                  {state.language === 'ar' ? 'نسبة العرض إلى الارتفاع:' : 'Aspect Ratio:'}
                </label>
                <select
                  id="videoAspectRatioSelect"
                  className="mt-1 block w-full pl-3 pr-10 py-2 text-base border-gray-300 focus:outline-none focus:ring-blue-500 focus:border-blue-500 sm:text-sm rounded-md"
                  value={videoGenAspectRatio}
                  onChange={(e) => setVideoGenAspectRatio(e.target.value as VideoAspectRatio)}
                >
                  {VIDEO_ASPECT_RATIOS.map((ratio) => (
                    <option key={ratio} value={ratio}>{ratio}</option>
                  ))}
                </select>
              </div>

              <div>
                <label htmlFor="videoResolutionSelect" className="block text-sm font-medium text-gray-700 mb-1">
                  {state.language === 'ar' ? 'الدقة:' : 'Resolution:'}
                </label>
                <select
                  id="videoResolutionSelect"
                  className="mt-1 block w-full pl-3 pr-10 py-2 text-base border-gray-300 focus:outline-none focus:ring-blue-500 focus:border-blue-500 sm:text-sm rounded-md"
                  value={videoGenResolution}
                  onChange={(e) => setVideoGenResolution(e.target.value as VideoResolution)}
                >
                  {VIDEO_RESOLUTIONS.map((res) => (
                    <option key={res} value={res}>{res}</option>
                  ))}
                </select>
              </div>

              <button
                onClick={handleVideoGeneration}
                className="w-full bg-blue-600 text-white py-3 px-6 rounded-md text-lg font-semibold hover:bg-blue-700 transition-colors duration-200 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2"
                disabled={state.loading || (!videoGenPrompt.trim() && !videoGenImageInput)}
              >
                {state.language === 'ar' ? 'توليد الفيديو' : 'Generate Video'}
              </button>

              {generatedVideoUrl && (
                <div className="mt-6 text-center">
                  <h3 className="text-xl font-semibold text-gray-800 mb-2">{state.language === 'ar' ? 'الفيديو المولّد:' : 'Generated Video:'}</h3>
                  <VideoPlayer src={generatedVideoUrl} controls autoplay loop />
                </div>
              )}
            </div>
          )}

          {/* Video Understanding Tab */}
          {state.activeTab === Tab.VIDEO_UNDERSTANDING && (
            <div className="space-y-6">
              <h2 className="text-2xl font-bold text-gray-800 mb-4">{state.language === 'ar' ? 'فهم الفيديو' : 'Video Understanding'}</h2>

              <ImageUploader
                label={state.language === 'ar' ? 'قم بتحميل الفيديو للتحليل (سيتم استخراج إطار واحد):' : 'Upload Video for Analysis (one frame will be extracted):'}
                allowedFileTypes={['video/mp4', 'video/webm', 'video/quicktime']}
                onFileChange={(file, imageInput) => {
                  setVideoUnderstandFile(file);
                  setVideoUnderstandFrame(imageInput); // This will be the extracted frame (ImageInput type)
                }}
              />

              <div>
                <label htmlFor="videoUnderstandPrompt" className="block text-sm font-medium text-gray-700 mb-1">
                  {state.language === 'ar' ? 'أدخل استفسارك عن الفيديو:' : 'Enter your prompt about the video:'}
                </label>
                <textarea
                  id="videoUnderstandPrompt"
                  className="w-full p-3 border border-gray-300 rounded-md focus:ring-blue-500 focus:border-blue-500 resize-y"
                  rows={4}
                  value={videoUnderstandPrompt}
                  onChange={(e) => setVideoUnderstandPrompt(e.target.value)}
                  placeholder={state.language === 'ar' ? 'مثال: صف المشهد الرئيسي في هذا الفيديو...' : 'e.g., Describe the main scene in this video...'}
                ></textarea>
              </div>

              <button
                onClick={handleVideoUnderstanding}
                className="w-full bg-blue-600 text-white py-3 px-6 rounded-md text-lg font-semibold hover:bg-blue-700 transition-colors duration-200 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2"
                disabled={state.loading || !videoUnderstandFrame || !videoUnderstandPrompt.trim()}
              >
                {state.language === 'ar' ? 'تحليل الفيديو' : 'Analyze Video'}
              </button>

              {videoUnderstandResponse && (
                <div className="mt-6 bg-gray-50 p-4 rounded-md shadow-inner">
                  <h3 className="text-xl font-semibold text-gray-800 mb-2">{state.language === 'ar' ? 'تحليل الفيديو:' : 'Video Analysis:'}</h3>
                  <div className="whitespace-pre-wrap text-gray-700">{videoUnderstandResponse}</div>
                </div>
              )}
            </div>
          )}

          {/* Live Chat Tab */}
          {state.activeTab === Tab.LIVE_CHAT && (
            <div className="space-y-6 flex flex-col h-full">
              <h2 className="text-2xl font-bold text-gray-800 mb-4">{state.language === 'ar' ? 'المحادثة المباشرة (صوت)' : 'Live Chat (Audio)'}</h2>

              <div>
                <label htmlFor="liveVoiceSelect" className="block text-sm font-medium text-gray-700 mb-1">
                  {state.language === 'ar' ? 'اختر الصوت:' : 'Select Voice:'}
                </label>
                <select
                  id="liveVoiceSelect"
                  className="mt-1 block w-full pl-3 pr-10 py-2 text-base border-gray-300 focus:outline-none focus:ring-blue-500 focus:border-blue-500 sm:text-sm rounded-md"
                  value={selectedVoice}
                  onChange={(e) => setSelectedVoice(e.target.value)}
                  disabled={isLiveChatActive}
                >
                  {VOICE_NAMES.map((voice) => (
                    <option key={voice.value} value={voice.value}>{voice.label}</option>
                  ))}
                </select>
              </div>

              <div className="flex space-x-4">
                <button
                  onClick={isLiveChatActive ? stopLiveChat : setupLiveChat}
                  className={`flex-1 py-3 px-6 rounded-md text-lg font-semibold transition-colors duration-200 focus:outline-none focus:ring-2 focus:ring-offset-2
                    ${isLiveChatActive
                      ? 'bg-red-600 hover:bg-red-700 text-white focus:ring-red-500'
                      : 'bg-green-600 hover:bg-green-700 text-white focus:ring-green-500'
                    }`}
                  disabled={state.loading && !isLiveChatActive}
                >
                  {isLiveChatActive
                    ? (state.language === 'ar' ? 'إيقاف المحادثة' : 'Stop Live Chat')
                    : (state.language === 'ar' ? 'بدء المحادثة المباشرة' : 'Start Live Chat')}
                </button>
              </div>

              <TranscriptionDisplay
                inputTranscription={liveChatInputTranscription}
                outputTranscription={liveChatOutputTranscription}
                history={liveChatHistory}
              />
            </div>
          )}

          {/* Text-to-Speech Tab */}
          {state.activeTab === Tab.TEXT_TO_SPEECH && (
            <div className="space-y-6">
              <h2 className="text-2xl font-bold text-gray-800 mb-4">{state.language === 'ar' ? 'تحويل النص إلى كلام' : 'Text-to-Speech'}</h2>

              <div>
                <label htmlFor="ttsVoiceSelect" className="block text-sm font-medium text-gray-700 mb-1">
                  {state.language === 'ar' ? 'اختر الصوت:' : 'Select Voice:'}
                </label>
                <select
                  id="ttsVoiceSelect"
                  className="mt-1 block w-full pl-3 pr-10 py-2 text-base border-gray-300 focus:outline-none focus:ring-blue-500 focus:border-blue-500 sm:text-sm rounded-md"
                  value={selectedVoice}
                  onChange={(e) => setSelectedVoice(e.target.value)}
                  disabled={state.loading}
                >
                  {VOICE_NAMES.map((voice) => (
                    <option key={voice.value} value={voice.value}>{voice.label}</option>
                  ))}
                </select>
              </div>

              <div>
                <label htmlFor="ttsInput" className="block text-sm font-medium text-gray-700 mb-1">
                  {state.language === 'ar' ? 'أدخل النص لتحويله إلى كلام:' : 'Enter text to convert to speech:'}
                </label>
                <textarea
                  id="ttsInput"
                  className="w-full p-3 border border-gray-300 rounded-md focus:ring-blue-500 focus:border-blue-500 resize-y"
                  rows={6}
                  value={ttsInput}
                  onChange={(e) => setTtsInput(e.target.value)}
                  placeholder={state.language === 'ar' ? 'مثال: مرحباً بك في تطبيق Gemini!' : 'e.g., Hello, welcome to the Gemini app!'}
                ></textarea>
              </div>

              <button
                onClick={handleTextToSpeech}
                className="w-full bg-blue-600 text-white py-3 px-6 rounded-md text-lg font-semibold hover:bg-blue-700 transition-colors duration-200 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2"
                disabled={state.loading || !ttsInput.trim()}
              >
                {state.language === 'ar' ? 'توليد الكلام وتشغيله' : 'Generate and Play Speech'}
              </button>

              <audio ref={audioRef} controls className="w-full mt-4"></audio>
            </div>
          )}

          {/* Chatbot Tab */}
          {state.activeTab === Tab.CHATBOT && (
            <div className="space-y-6 flex flex-col h-[70vh]">
              <h2 className="text-2xl font-bold text-gray-800 mb-4">{state.language === 'ar' ? 'روبوت الدردشة' : 'Chatbot'}</h2>

              <div className="flex-1 bg-gray-50 p-4 rounded-lg shadow-inner overflow-y-auto flex flex-col space-y-3">
                {chatMessages.length === 0 && (
                  <p className="text-gray-500 text-center italic">
                    {state.language === 'ar' ? 'ابدأ الدردشة مع مساعد Gemini.' : 'Start a conversation with the Gemini assistant.'}
                  </p>
                )}
                {chatMessages.map((msg, index) => (
                  <div
                    key={index}
                    className={`p-3 rounded-lg max-w-[80%] ${
                      msg.role === 'user'
                        ? 'bg-blue-100 text-blue-800 self-end ' + (state.language === 'ar' ? 'text-right' : 'text-left')
                        : 'bg-green-100 text-green-800 self-start ' + (state.language === 'ar' ? 'text-right' : 'text-left')
                    }`}
                  >
                    <strong>{msg.role === 'user' ? (state.language === 'ar' ? 'أنت:' : 'You:') : (state.language === 'ar' ? 'جميني:' : 'Gemini:')}</strong> {msg.content}
                  </div>
                ))}
                {state.loading && (
                  <div className={`self-start ${state.language === 'ar' ? 'text-right' : 'text-left'}`}>
                    <LoadingSpinner message={state.language === 'ar' ? 'جميني يكتب...' : 'Gemini is typing...'} className="!p-0 !h-auto !w-auto" />
                  </div>
                )}
              </div>

              <div className="flex gap-2 mt-4 sticky bottom-0 bg-white pt-4">
                <textarea
                  ref={chatInputRef}
                  className="flex-1 p-3 border border-gray-300 rounded-md focus:ring-blue-500 focus:border-blue-500 resize-none"
                  rows={2}
                  placeholder={state.language === 'ar' ? 'اكتب رسالتك هنا...' : 'Type your message here...'}
                  onKeyPress={(e) => {
                    if (e.key === 'Enter' && !e.shiftKey) {
                      e.preventDefault();
                      handleChatMessageSend();
                    }
                  }}
                  disabled={state.loading}
                ></textarea>
                <button
                  onClick={handleChatMessageSend}
                  className="bg-blue-600 text-white py-2 px-4 rounded-md text-lg font-semibold hover:bg-blue-700 transition-colors duration-200 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2"
                  disabled={state.loading}
                >
                  {state.language === 'ar' ? 'إرسال' : 'Send'}
                </button>
              </div>
            </div>
          )}

          {/* Grounding Tab */}
          {state.activeTab === Tab.GROUNDING && (
            <div className="space-y-6">
              <h2 className="text-2xl font-bold text-gray-800 mb-4">{state.language === 'ar' ? 'التأريض (Search & Maps)' : 'Grounding (Search & Maps)'}</h2>

              <div>
                <label htmlFor="groundingPrompt" className="block text-sm font-medium text-gray-700 mb-1">
                  {state.language === 'ar' ? 'أدخل استفسارك (على سبيل المثال، "من فاز بأكبر عدد من الميداليات الذهبية في أولمبياد باريس 2024؟" أو "ما هي المطاعم الإيطالية الجيدة القريبة؟"):' : 'Enter your query (e.g., "Who won the most gold medals at the Paris Olympics in 2024?" or "What good Italian restaurants are nearby?"): '}
                </label>
                <textarea
                  id="groundingPrompt"
                  className="w-full p-3 border border-gray-300 rounded-md focus:ring-blue-500 focus:border-blue-500 resize-y"
                  rows={4}
                  value={groundingPrompt}
                  onChange={(e) => setGroundingPrompt(e.target.value)}
                  placeholder={state.language === 'ar' ? 'أدخل استفسارك هنا...' : 'Enter your query here...'}
                ></textarea>
              </div>

              <div className="flex items-center space-x-4">
                <label className="flex items-center">
                  <input
                    type="checkbox"
                    checked={groundingTools.googleSearch}
                    onChange={(e) => setGroundingTools(prev => ({ ...prev, googleSearch: e.target.checked }))}
                    className="h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 rounded"
                  />
                  <span className={`ml-2 text-sm font-medium text-gray-700 ${state.language === 'ar' ? 'mr-2' : ''}`}>
                    {state.language === 'ar' ? 'بحث جوجل' : 'Google Search'}
                  </span>
                </label>
                <label className="flex items-center">
                  <input
                    type="checkbox"
                    checked={groundingTools.googleMaps}
                    onChange={(e) => setGroundingTools(prev => ({ ...prev, googleMaps: e.target.checked }))}
                    className="h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 rounded"
                  />
                  <span className={`ml-2 text-sm font-medium text-gray-700 ${state.language === 'ar' ? 'mr-2' : ''}`}>
                    {state.language === 'ar' ? 'خرائط جوجل' : 'Google Maps'}
                  </span>
                </label>
              </div>

              {groundingTools.googleMaps && !geolocation && !state.loading && (
                <ErrorMessage message={state.language === 'ar' ? 'جارٍ الحصول على موقعك... يرجى السماح بالوصول إلى الموقع.' : 'Getting your location... Please allow location access.'} />
              )}
              {groundingTools.googleMaps && geolocation && (
                <p className="text-sm text-green-700">
                  {state.language === 'ar' ? `تم الحصول على الموقع: خط العرض ${geolocation.coords.latitude.toFixed(4)}, خط الطول ${geolocation.coords.longitude.toFixed(4)}` : `Location obtained: Lat ${geolocation.coords.latitude.toFixed(4)}, Lng ${geolocation.coords.longitude.toFixed(4)}`}
                </p>
              )}


              <button
                onClick={handleGroundingSearch}
                className="w-full bg-blue-600 text-white py-3 px-6 rounded-md text-lg font-semibold hover:bg-blue-700 transition-colors duration-200 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2"
                disabled={state.loading || !groundingPrompt.trim() || (groundingTools.googleMaps && !geolocation)}
              >
                {state.language === 'ar' ? 'تشغيل البحث' : 'Run Grounded Search'}
              </button>

              {groundingResponse && (
                <div className="mt-6 bg-gray-50 p-4 rounded-md shadow-inner">
                  <h3 className="text-xl font-semibold text-gray-800 mb-2">{state.language === 'ar' ? 'الاستجابة:' : 'Response:'}</h3>
                  <div className="whitespace-pre-wrap text-gray-700">{groundingResponse}</div>

                  {groundingLinks.length > 0 && (
                    <div className="mt-4">
                      <h4 className="text-lg font-semibold text-gray-800 mb-2">{state.language === 'ar' ? 'المصادر:' : 'Sources:'}</h4>
                      <ul className="list-disc pl-5 space-y-1">
                        {groundingLinks.map((chunk, index) => (
                          <li key={index}>
                            {chunk.web && chunk.web.uri && ( // Ensure uri exists before rendering link
                              <a href={chunk.web.uri} target="_blank" rel="noopener noreferrer" className="text-blue-500 hover:underline">
                                {chunk.web.title || chunk.web.uri}
                              </a>
                            )}
                            {chunk.maps && chunk.maps.uri && ( // Ensure uri exists before rendering link
                              <div>
                                <a href={chunk.maps.uri} target="_blank" rel="noopener noreferrer" className="text-blue-500 hover:underline">
                                  {chunk.maps.title || chunk.maps.uri}
                                </a>
                                {chunk.maps.placeAnswerSources?.reviewSnippets?.map((snippet, sIdx) => (
                                  <p key={sIdx} className="ml-4 text-sm text-gray-600 italic">
                                    "{snippet.reviewSnippet}" - <a href={snippet.uri} target="_blank" rel="noopener noreferrer" className="text-blue-400 hover:underline">Review Link</a>
                                  </p>
                                ))}
                              </div>
                            )}
                          </li>
                        ))}
                      </ul>
                    </div>
                  )}
                </div>
              )}
            </div>
          )}

        </Tabs>
      </div>
    </div>
  );
};

export default App;