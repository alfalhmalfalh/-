import React, { useState, useRef, useCallback, useEffect } from 'react';
import { ImageInput } from '../types';

interface ImageUploaderProps {
  onFileChange: (file: File | null, imageInput?: ImageInput | null) => void;
  allowedFileTypes: string[];
  label: string;
  className?: string;
  multiple?: boolean;
}

const ImageUploader: React.FC<ImageUploaderProps> = ({
  onFileChange,
  allowedFileTypes,
  label,
  className,
  multiple = false,
}) => {
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  const readFileAsDataURL = (file: File): Promise<string> => {
    return new Promise((resolve) => {
      const reader = new FileReader();
      reader.onloadend = () => {
        resolve(reader.result as string);
      };
      reader.readAsDataURL(file);
    });
  };

  const extractVideoFrame = useCallback(async (videoFile: File): Promise<ImageInput> => {
    return new Promise((resolve, reject) => {
      const video = document.createElement('video');
      video.autoplay = true;
      video.muted = true;
      video.src = URL.createObjectURL(videoFile);

      video.onloadedmetadata = () => {
        video.currentTime = Math.min(1, video.duration / 2); // Get a frame near the beginning
      };

      video.onseeked = () => {
        const canvas = document.createElement('canvas');
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        const ctx = canvas.getContext('2d');
        if (ctx) {
          ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
          canvas.toBlob(
            (blob) => {
              if (blob) {
                const reader = new FileReader();
                reader.onloadend = () => {
                  const base64 = (reader.result as string).split(',')[1];
                  resolve({ base64Data: base64, mimeType: 'image/jpeg' });
                  URL.revokeObjectURL(video.src);
                };
                reader.readAsDataURL(blob);
              } else {
                reject(new Error('Canvas to Blob failed.'));
              }
            },
            'image/jpeg',
            0.9
          );
        } else {
          reject(new Error('Could not get 2D context for canvas.'));
        }
      };

      video.onerror = (e) => {
        reject(new Error(`Error loading video: ${e}`));
      };
    });
  }, []);


  const handleFileChange = useCallback(async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      const isImage = file.type.startsWith('image/');
      const isVideo = file.type.startsWith('video/');

      if (isImage) {
        const dataUrl = await readFileAsDataURL(file);
        setPreviewUrl(dataUrl);
        onFileChange(file, { base64Data: dataUrl.split(',')[1], mimeType: file.type });
      } else if (isVideo) {
        const objectUrl = URL.createObjectURL(file);
        setPreviewUrl(objectUrl); // Show video preview
        try {
          const imageInput = await extractVideoFrame(file); // Extract frame for Gemini
          onFileChange(file, imageInput);
        } catch (error) {
          console.error('Error extracting video frame:', error);
          onFileChange(file, null); // Still provide the file, but no image input
        }
      } else {
        setPreviewUrl(null);
        onFileChange(null);
        alert('Unsupported file type.');
      }
    } else {
      setPreviewUrl(null);
      onFileChange(null);
    }
  }, [onFileChange, extractVideoFrame]);

  useEffect(() => {
    return () => {
      if (previewUrl && previewUrl.startsWith('blob:')) {
        URL.revokeObjectURL(previewUrl);
      }
    };
  }, [previewUrl]);

  const handleClear = useCallback(() => {
    setPreviewUrl(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
    onFileChange(null);
  }, [onFileChange]);

  return (
    <div className={`flex flex-col items-center border border-gray-300 rounded-lg p-4 space-y-4 ${className}`}>
      <label className="block text-sm font-medium text-gray-700">{label}</label>
      <input
        type="file"
        accept={allowedFileTypes.join(',')}
        onChange={handleFileChange}
        ref={fileInputRef}
        multiple={multiple}
        className="block w-full text-sm text-gray-500
                   file:mr-4 file:py-2 file:px-4
                   file:rounded-full file:border-0
                   file:text-sm file:font-semibold
                   file:bg-blue-50 file:text-blue-700
                   hover:file:bg-blue-100 cursor-pointer"
      />
      {previewUrl && (
        <div className="mt-4 w-full max-w-sm border border-gray-200 rounded-lg overflow-hidden shadow-sm">
          {previewUrl.startsWith('data:image/') ? (
            <img src={previewUrl} alt="Preview" className="w-full h-auto object-contain" />
          ) : (
            <video ref={videoRef} src={previewUrl} controls className="w-full h-auto object-contain"></video>
          )}
          <button
            onClick={handleClear}
            className="w-full py-2 bg-red-500 text-white rounded-b-lg hover:bg-red-600 transition-colors duration-200"
          >
            Clear File
          </button>
        </div>
      )}
      <canvas ref={canvasRef} className="hidden"></canvas> {/* Hidden canvas for video frame extraction */}
    </div>
  );
};

export default ImageUploader;
