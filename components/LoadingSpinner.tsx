import React from 'react';

interface LoadingSpinnerProps {
  message?: string;
  className?: string;
}

const LoadingSpinner: React.FC<LoadingSpinnerProps> = ({ message, className }) => {
  return (
    <div className={`flex flex-col items-center justify-center p-4 ${className}`}>
      <div className="animate-spin rounded-full h-12 w-12 border-4 border-t-4 border-blue-500 border-opacity-25 border-t-blue-500"></div>
      {message && <p className="mt-3 text-lg text-gray-700">{message}</p>}
    </div>
  );
};

export default LoadingSpinner;