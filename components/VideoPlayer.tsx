import React from 'react';

interface VideoPlayerProps {
  src: string;
  className?: string;
  autoplay?: boolean;
  controls?: boolean;
  loop?: boolean;
  muted?: boolean;
}

const VideoPlayer: React.FC<VideoPlayerProps> = ({
  src,
  className,
  autoplay = false,
  controls = true,
  loop = false,
  muted = false,
}) => {
  return (
    <video
      src={src}
      className={`w-full max-w-md h-auto rounded-lg shadow-lg ${className}`}
      autoPlay={autoplay}
      controls={controls}
      loop={loop}
      muted={muted}
      playsInline
    >
      Your browser does not support the video tag.
    </video>
  );
};

export default VideoPlayer;
