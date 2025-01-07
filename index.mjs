import fs from 'fs';
import { exec } from 'child_process';
import recorder from 'node-record-lpcm16';

const recordAudio = () => {
  return new Promise((resolve, reject) => {
    console.log('Recording audio...');
    const audioChunks = [];

    const recording = recorder.record({
      sampleRate: 16000,
      channels: 1,
      audioType: 'wav',
      // Add these options to specify the recorder and device
      recorder: 'sox',
      device: 0
    });

    recording.stream()
      .on('data', (chunk) => {
        audioChunks.push(chunk);
      })
      .on('error', (err) => {
        console.error('Recording error:', err);
        reject(err);
      });

    // Record for 5 seconds
    setTimeout(() => {
      recording.stop();
      const audioBuffer = Buffer.concat(audioChunks);
      const fileName = 'audio.wav';
      fs.writeFile(fileName, audioBuffer, (err) => {
        if (err) {
          console.error('Error saving audio file:', err);
          reject(err);
        } else {
          console.log('Audio recording saved as audio.wav');
          resolve(fileName);
        }
      });
    }, 5000);
  });
};

const transcribeAudio = (fileName) => {
  return new Promise((resolve, reject) => {
    exec(`python transcribe.py ${fileName}`, (error, stdout, stderr) => {
      if (error) {
        console.error(`Error executing Whisper: ${error.message}`);
        reject(error);
        return;
      }
      if (stderr) {
        console.error(`Error during transcription: ${stderr}`);
        reject(new Error(stderr));
        return;
      }

      resolve(stdout.trim());
    });
  });
};

(async () => {
  try {
    console.log('Starting recording process...');
    const fileName = await recordAudio();
    console.log('Recording completed successfully.');

    console.log('Starting transcription...');
    const transcription = await transcribeAudio(fileName);
    console.log('Transcription completed.');
    console.log('Transcription result:');
    console.log(transcription);

    // You can return or further process the transcription here
    return transcription;
  } catch (error) {
    console.error('Error during audio recording or transcription process:', error);
  }
})();