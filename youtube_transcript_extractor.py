from youtube_transcript_api import YouTubeTranscriptApi

class YoutubeTranscriptExtractor:
    def extract_transcript(self, video_url):
        try:
            video_id = video_url.split("watch?v=")[1]
            transcript = YouTubeTranscriptApi.get_transcript(video_id)
            text = " ".join([entry['text'] for entry in transcript])
            return text
        except Exception as e:
            return f"Error extracting transcript from {video_url}: {e}"

    def __call__(self, video_url): # Makes the class callable like a function.  Important for tools
        return self.extract_transcript(video_url)
