import React, { useRef, useEffect } from 'react';

interface TranscriptionDisplayProps {
  inputTranscription: string;
  outputTranscription: string;
  history: string[];
}

const TranscriptionDisplay: React.FC<TranscriptionDisplayProps> = ({
  inputTranscription,
  outputTranscription,
  history,
}) => {
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [inputTranscription, outputTranscription, history]);

  return (
    <div className="bg-gray-50 p-4 rounded-lg shadow-inner flex-1 max-h-96 overflow-y-auto" ref={scrollRef}>
      <h3 className="text-lg font-semibold mb-2 text-gray-800">Conversation Transcription:</h3>
      <div className="space-y-3">
        {history.map((turn, index) => (
          <p key={index} className={`text-gray-600 ${index % 2 === 0 ? 'text-blue-700' : 'text-green-700'}`}>
            <strong>{index % 2 === 0 ? 'User:' : 'AI:'}</strong> {turn}
          </p>
        ))}
        {inputTranscription && (
          <p className="text-blue-700"><strong>User (Live):</strong> {inputTranscription}</p>
        )}
        {outputTranscription && (
          <p className="text-green-700"><strong>AI (Live):</strong> {outputTranscription}</p>
        )}
      </div>
    </div>
  );
};

export default TranscriptionDisplay;
