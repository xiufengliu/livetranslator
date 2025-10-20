import React, { useCallback, useEffect, useRef, useState } from 'react';
import ReactDOM from 'react-dom/client';
import { GoogleGenAI, LiveServerMessage, Modality, Blob } from '@google/genai';

function decode(base64: string) {
  const binaryString = atob(base64);
  const len = binaryString.length;
  const bytes = new Uint8Array(len);
  for (let i = 0; i < len; i += 1) {
    bytes[i] = binaryString.charCodeAt(i);
  }
  return bytes;
}

function encode(bytes: Uint8Array) {
  let binary = '';
  const len = bytes.byteLength;
  for (let i = 0; i < len; i += 1) {
    binary += String.fromCharCode(bytes[i]);
  }
  return btoa(binary);
}

async function decodeAudioData(
  data: Uint8Array,
  ctx: AudioContext,
  sampleRate: number,
  numChannels: number,
): Promise<AudioBuffer> {
  const dataInt16 = new Int16Array(data.buffer);
  const frameCount = dataInt16.length / numChannels;
  const buffer = ctx.createBuffer(numChannels, frameCount, sampleRate);

  for (let channel = 0; channel < numChannels; channel += 1) {
    const channelData = buffer.getChannelData(channel);
    for (let i = 0; i < frameCount; i += 1) {
      channelData[i] = dataInt16[i * numChannels + channel] / 32768;
    }
  }
  return buffer;
}

const LANGUAGE_PAIRS = [
  // 中文（简体） ⇄ 主要语言
  { name: '中文（简体） → 英语', source: '中文（简体）', target: '英语' },
  { name: '英语 → 中文（简体）', source: '英语', target: '中文（简体）' },
  { name: '中文（简体） → 丹麦语', source: '中文（简体）', target: '丹麦语' },
  { name: '丹麦语 → 中文（简体）', source: '丹麦语', target: '中文（简体）' },
  { name: '中文（简体） → 法语', source: '中文（简体）', target: '法语' },
  { name: '法语 → 中文（简体）', source: '法语', target: '中文（简体）' },
  { name: '中文（简体） → 德语', source: '中文（简体）', target: '德语' },
  { name: '德语 → 中文（简体）', source: '德语', target: '中文（简体）' },
  { name: '中文（简体） → 西班牙语', source: '中文（简体）', target: '西班牙语' },
  { name: '西班牙语 → 中文（简体）', source: '西班牙语', target: '中文（简体）' },
  { name: '中文（简体） → 葡萄牙语', source: '中文（简体）', target: '葡萄牙语' },
  { name: '葡萄牙语 → 中文（简体）', source: '葡萄牙语', target: '中文（简体）' },
  { name: '中文（简体） → 意大利语', source: '中文（简体）', target: '意大利语' },
  { name: '意大利语 → 中文（简体）', source: '意大利语', target: '中文（简体）' },
  { name: '中文（简体） → 日语', source: '中文（简体）', target: '日语' },
  { name: '日语 → 中文（简体）', source: '日语', target: '中文（简体）' },
  { name: '中文（简体） → 韩语', source: '中文（简体）', target: '韩语' },
  { name: '韩语 → 中文（简体）', source: '韩语', target: '中文（简体）' },
  { name: '中文（简体） → 俄语', source: '中文（简体）', target: '俄语' },
  { name: '俄语 → 中文（简体）', source: '俄语', target: '中文（简体）' },
  { name: '中文（简体） → 阿拉伯语', source: '中文（简体）', target: '阿拉伯语' },
  { name: '阿拉伯语 → 中文（简体）', source: '阿拉伯语', target: '中文（简体）' },
  { name: '中文（简体） → 印地语', source: '中文（简体）', target: '印地语' },
  { name: '印地语 → 中文（简体）', source: '印地语', target: '中文（简体）' },
  { name: '中文（简体） → 泰语', source: '中文（简体）', target: '泰语' },
  { name: '泰语 → 中文（简体）', source: '泰语', target: '中文（简体）' },
  { name: '中文（简体） → 越南语', source: '中文（简体）', target: '越南语' },
  { name: '越南语 → 中文（简体）', source: '越南语', target: '中文（简体）' },
  { name: '中文（简体） → 印度尼西亚语', source: '中文（简体）', target: '印度尼西亚语' },
  { name: '印度尼西亚语 → 中文（简体）', source: '印度尼西亚语', target: '中文（简体）' },
  { name: '中文（简体） → 土耳其语', source: '中文（简体）', target: '土耳其语' },
  { name: '土耳其语 → 中文（简体）', source: '土耳其语', target: '中文（简体）' },
  { name: '中文（简体） → 荷兰语', source: '中文（简体）', target: '荷兰语' },
  { name: '荷兰语 → 中文（简体）', source: '荷兰语', target: '中文（简体）' },
  { name: '中文（简体） → 波兰语', source: '中文（简体）', target: '波兰语' },
  { name: '波兰语 → 中文（简体）', source: '波兰语', target: '中文（简体）' },
  { name: '中文（简体） → 瑞典语', source: '中文（简体）', target: '瑞典语' },
  { name: '瑞典语 → 中文（简体）', source: '瑞典语', target: '中文（简体）' },
  { name: '中文（简体） → 挪威语', source: '中文（简体）', target: '挪威语' },
  { name: '挪威语 → 中文（简体）', source: '挪威语', target: '中文（简体）' },
  { name: '中文（简体） → 芬兰语', source: '中文（简体）', target: '芬兰语' },
  { name: '芬兰语 → 中文（简体）', source: '芬兰语', target: '中文（简体）' },
];

const FALLBACK_PAIR = LANGUAGE_PAIRS[0];
const MAX_TRANSCRIPT_ENTRIES = 200;

type TranscriptEntry = {
  sourceText: string;
  translatedText: string;
  sourceLang: string;
  targetLang: string;
  createdAt: number;
};

const timeFormatter = new Intl.DateTimeFormat(undefined, {
  hour: '2-digit',
  minute: '2-digit',
});
const formatTimestamp = (value: number) => timeFormatter.format(value);

const TRANSCRIPT_KEY = 'liveTranslatorTranscript';
const LANGUAGE_PAIR_KEY = 'selectedPairIndex';

const loadTranscriptFromStorage = (): TranscriptEntry[] => {
  try {
    const saved = localStorage.getItem(TRANSCRIPT_KEY);
    if (!saved) return [];
    const parsed = JSON.parse(saved);
    if (!Array.isArray(parsed)) return [];

    return parsed
      .map((entry: any) => {
        if (!entry || typeof entry !== 'object') return null;
        const sourceText = typeof entry.sourceText === 'string' ? entry.sourceText : '';
        const translatedText = typeof entry.translatedText === 'string' ? entry.translatedText : '';
        const sourceLang =
          typeof entry.sourceLang === 'string' ? entry.sourceLang : FALLBACK_PAIR.source;
        const targetLang =
          typeof entry.targetLang === 'string' ? entry.targetLang : FALLBACK_PAIR.target;
        const createdAt = typeof entry.createdAt === 'number' ? entry.createdAt : Date.now();
        if (!sourceText && !translatedText) return null;
        return { sourceText, translatedText, sourceLang, targetLang, createdAt };
      })
      .filter(Boolean) as TranscriptEntry[];
  } catch (error) {
    console.error('Failed to parse transcript from localStorage', error);
    return [];
  }
};

const LiveTranslatorApp = () => {
  const [transcript, setTranscript] = useState<TranscriptEntry[]>(loadTranscriptFromStorage);
  const [isRecording, setIsRecording] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [isMaximized, setIsMaximized] = useState(false);
  const [isSettingsOpen, setIsSettingsOpen] = useState(false);
  const [status, setStatus] = useState<'idle' | 'connecting' | 'listening'>('idle');
  const [selectedPairIndex, setSelectedPairIndex] = useState(() => {
    const saved = localStorage.getItem(LANGUAGE_PAIR_KEY);
    const parsed = saved ? Number.parseInt(saved, 10) : 0;
    if (Number.isNaN(parsed)) return 0;
    return Math.min(Math.max(parsed, 0), LANGUAGE_PAIRS.length - 1);
  });
  const selectedPair = LANGUAGE_PAIRS[selectedPairIndex] ?? FALLBACK_PAIR;

  const transcriptRef = useRef<HTMLDivElement>(null);
  const sessionPromiseRef = useRef<Promise<any> | null>(null);
  const sessionRef = useRef<any>(null);
  const mediaStreamRef = useRef<MediaStream | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const outputAudioContextRef = useRef<AudioContext | null>(null);
  const outputGainNodeRef = useRef<GainNode | null>(null);
  const scriptProcessorRef = useRef<ScriptProcessorNode | null>(null);
  const mediaStreamSourceRef = useRef<MediaStreamAudioSourceNode | null>(null);
  const outputSourcesRef = useRef<Set<AudioBufferSourceNode>>(new Set());
  const nextOutputStartTimeRef = useRef(0);
  const isRecordingRef = useRef(false);
  const statusRef = useRef<'idle' | 'connecting' | 'listening'>('idle');
  const isMountedRef = useRef(true);
  // Discrete speech pace control: slow | normal | fast
  type Pace = 'slow' | 'normal' | 'fast';
  const SPEECH_PACE_KEY = 'speechPace';
  const [pace, setPace] = useState<Pace>(() => {
    const saved = localStorage.getItem(SPEECH_PACE_KEY);
    return saved === 'slow' || saved === 'fast' || saved === 'normal' ? (saved as Pace) : 'normal';
  });
  const paceRef = useRef<Pace>(pace);
  useEffect(() => {
    paceRef.current = pace;
  }, [pace]);

  // Keep the screen awake while recording so the OS doesn't lock
  const wakeLockRef = useRef<any>(null);
  const requestWakeLock = useCallback(async () => {
    try {
      const anyNavigator: any = navigator as any;
      if (anyNavigator?.wakeLock?.request) {
        wakeLockRef.current = await anyNavigator.wakeLock.request('screen');
      }
    } catch (e) {
      console.debug('Wake Lock not available or denied', e);
    }
  }, []);
  const releaseWakeLock = useCallback(async () => {
    try {
      if (wakeLockRef.current?.release) {
        await wakeLockRef.current.release();
      }
    } catch {}
    wakeLockRef.current = null;
  }, []);

  const currentInputTranscription = useRef('');
  const currentOutputTranscription = useRef('');

  useEffect(() => {
    isRecordingRef.current = isRecording;
  }, [isRecording]);

  useEffect(() => {
    statusRef.current = status;
  }, [status]);

  useEffect(() => {
    if (transcriptRef.current) {
      transcriptRef.current.scrollTop = transcriptRef.current.scrollHeight;
    }
  }, [transcript]);

  useEffect(() => {
    localStorage.setItem(TRANSCRIPT_KEY, JSON.stringify(transcript));
  }, [transcript]);

  useEffect(() => {
    localStorage.setItem(LANGUAGE_PAIR_KEY, selectedPairIndex.toString());
  }, [selectedPairIndex]);
  useEffect(() => {
    localStorage.setItem(SPEECH_PACE_KEY, pace);
  }, [pace]);

  useEffect(() => {
    if (isMaximized || isSettingsOpen) {
      document.body.classList.add('overlay-open');
    } else {
      document.body.classList.remove('overlay-open');
    }
    return () => {
      document.body.classList.remove('overlay-open');
    };
  }, [isMaximized, isSettingsOpen]);

  // Reacquire wake lock on visibility if still recording
  useEffect(() => {
    const onVisibility = () => {
      if (document.visibilityState === 'visible' && isRecordingRef.current) {
        requestWakeLock();
      }
    };
    document.addEventListener('visibilitychange', onVisibility);
    return () => document.removeEventListener('visibilitychange', onVisibility);
  }, [requestWakeLock]);

  const clearTranscript = useCallback(() => {
    setTranscript([]);
    try {
      localStorage.removeItem(TRANSCRIPT_KEY);
    } catch (storageError) {
      console.error('Failed to clear transcript storage', storageError);
    }
  }, []);

  const stopRecording = useCallback(
    ({ skipStateReset = false }: { skipStateReset?: boolean } = {}) => {
      const sessionPromise = sessionPromiseRef.current;
      const activeSession = sessionRef.current;

      if (!isRecordingRef.current && !sessionPromise && !activeSession) {
        return;
      }

      // Release wake lock when stopping
      releaseWakeLock();

      sessionPromiseRef.current = null;
      sessionRef.current = null;

      if (sessionPromise) {
        sessionPromise
          .then((session: any) => {
            try {
              session?.close?.();
            } catch (closeError) {
              console.error('Failed to close session', closeError);
            }
          })
          .catch((resolveError: unknown) => {
            console.error('Error resolving live session during stop', resolveError);
          });
      } else if (activeSession) {
        try {
          activeSession.close?.();
        } catch (closeError) {
          console.error('Failed to close live session', closeError);
        }
      }

      if (mediaStreamRef.current) {
        mediaStreamRef.current.getTracks().forEach((track) => track.stop());
        mediaStreamRef.current = null;
      }

      if (scriptProcessorRef.current) {
        scriptProcessorRef.current.disconnect();
        scriptProcessorRef.current.onaudioprocess = null;
        scriptProcessorRef.current = null;
      }

      if (mediaStreamSourceRef.current) {
        mediaStreamSourceRef.current.disconnect();
        mediaStreamSourceRef.current = null;
      }

      if (audioContextRef.current && audioContextRef.current.state !== 'closed') {
        audioContextRef.current.close().catch((closeError) => {
          console.error('Failed to close input audio context', closeError);
        });
      }
      audioContextRef.current = null;

      outputSourcesRef.current.forEach((source) => {
        try {
          source.stop();
        } catch (stopError) {
          console.error('Failed to stop output audio source', stopError);
        }
      });
      outputSourcesRef.current.clear();
      nextOutputStartTimeRef.current = 0;

      if (outputGainNodeRef.current) {
        outputGainNodeRef.current.disconnect();
        outputGainNodeRef.current = null;
      }

      if (outputAudioContextRef.current && outputAudioContextRef.current.state !== 'closed') {
        outputAudioContextRef.current.close().catch((closeError) => {
          console.error('Failed to close output audio context', closeError);
        });
      }
      outputAudioContextRef.current = null;

      currentInputTranscription.current = '';
      currentOutputTranscription.current = '';

      isRecordingRef.current = false;
      statusRef.current = 'idle';

      if (!skipStateReset && isMountedRef.current) {
        setIsRecording(false);
        setStatus('idle');
      }
    },
    [],
  );

  useEffect(() => {
    return () => {
      isMountedRef.current = false;
      stopRecording({ skipStateReset: true });
    };
  }, [stopRecording]);

  const handlePairChange = useCallback(
    (index: number) => {
      const boundedIndex = Math.min(Math.max(index, 0), LANGUAGE_PAIRS.length - 1);
      if (boundedIndex === selectedPairIndex) return;

      if (isRecordingRef.current || statusRef.current !== 'idle') {
        stopRecording();
      }
      setSelectedPairIndex(boundedIndex);
    },
    [selectedPairIndex, stopRecording],
  );

  const startRecording = useCallback(async () => {
    if (isRecordingRef.current || sessionPromiseRef.current) {
      return;
    }

    try {
      setError(null);
      statusRef.current = 'connecting';
      if (isMountedRef.current) {
        setStatus('connecting');
      }

      const pair = LANGUAGE_PAIRS[selectedPairIndex] ?? FALLBACK_PAIR;
      const paceHint =
        paceRef.current === 'slow'
          ? 'Speak slowly and clearly to aid comprehension.'
          : paceRef.current === 'fast'
          ? 'Speak a bit faster while remaining clear.'
          : 'Use a natural speaking pace.';
      const systemInstruction = `You are an expert simultaneous interpreter. The user will speak to you in ${pair.source}. Your task is to listen to the user and provide a real-time, accurate translation in ${pair.target}. ONLY output the translation. Do not add any extra commentary, greetings, or explanations. Your response must be the direct translation of the user's speech. ${paceHint}`;

      const OutputContextCtor = window.AudioContext || (window as any).webkitAudioContext;
      const outputAudioContext = new OutputContextCtor({ sampleRate: 24000 });
      outputAudioContextRef.current = outputAudioContext;
      const outputGain = outputAudioContext.createGain();
      outputGain.gain.value = 1;
      outputGain.connect(outputAudioContext.destination);
      outputGainNodeRef.current = outputGain;

      outputSourcesRef.current.forEach((source) => {
        try {
          source.stop();
        } catch (stopError) {
          console.error('Failed to stop residual audio source', stopError);
        }
      });
      outputSourcesRef.current.clear();
      nextOutputStartTimeRef.current = 0;

      const apiKey = (process.env.API_KEY as unknown as string) || '';
      if (!apiKey) {
        setError('Missing Gemini API key. Please configure GEMINI_API_KEY on the server and refresh.');
        return;
      }

      const ai = new GoogleGenAI({ apiKey });
      sessionPromiseRef.current = ai.live.connect({
        model: 'gemini-2.5-flash-native-audio-preview-09-2025',
        callbacks: {
          onopen: () => {
            if (!isMountedRef.current) return;
            statusRef.current = 'listening';
            setStatus('listening');
          },
          onmessage: async (message: LiveServerMessage) => {
            if (message.serverContent?.outputTranscription?.text) {
              currentOutputTranscription.current += message.serverContent.outputTranscription.text;
            } else if (message.serverContent?.inputTranscription?.text) {
              currentInputTranscription.current += message.serverContent.inputTranscription.text;
            }

            if (message.serverContent?.turnComplete) {
              const sourceText = currentInputTranscription.current.trim();
              const translatedText = currentOutputTranscription.current.trim();

              if (sourceText || translatedText) {
                setTranscript((prev) => {
                  const nextEntry: TranscriptEntry = {
                    sourceText,
                    translatedText,
                    sourceLang: pair.source,
                    targetLang: pair.target,
                    createdAt: Date.now(),
                  };
                  const nextTranscript = [...prev, nextEntry];
                  if (nextTranscript.length > MAX_TRANSCRIPT_ENTRIES) {
                    return nextTranscript.slice(nextTranscript.length - MAX_TRANSCRIPT_ENTRIES);
                  }
                  return nextTranscript;
                });
              }

              currentInputTranscription.current = '';
              currentOutputTranscription.current = '';
            }

            const base64Audio = message.serverContent?.modelTurn?.parts?.[0]?.inlineData?.data;
            if (base64Audio) {
              const outputCtx = outputAudioContextRef.current;
              const outputNode = outputGainNodeRef.current;
              if (outputCtx && outputNode) {
                try {
                  const buffer = await decodeAudioData(decode(base64Audio), outputCtx, 24000, 1);
                  const source = outputCtx.createBufferSource();
                  source.buffer = buffer;
                  source.connect(outputNode);

                  const startAt = Math.max(nextOutputStartTimeRef.current, outputCtx.currentTime);
                  source.start(startAt);
                  nextOutputStartTimeRef.current = startAt + buffer.duration;

                  source.addEventListener('ended', () => {
                    outputSourcesRef.current.delete(source);
                  });
                  outputSourcesRef.current.add(source);
                } catch (playbackError) {
                  console.error('Failed to play translated audio', playbackError);
                }
              }
            }

            if (message.serverContent?.interrupted) {
              outputSourcesRef.current.forEach((source) => {
                try {
                  source.stop();
                } catch (stopError) {
                  console.error('Failed to stop interrupted audio source', stopError);
                }
              });
              outputSourcesRef.current.clear();
              nextOutputStartTimeRef.current = 0;
            }
          },
          onerror: (event: ErrorEvent) => {
            console.error('Session error:', event);
            if (isMountedRef.current) {
              setError('Connection error. Please try again.');
            }
            stopRecording();
          },
          onclose: () => {
            if (!isMountedRef.current) return;
            if (statusRef.current !== 'idle' || isRecordingRef.current) {
              stopRecording();
            }
          },
        },
        config: {
          responseModalities: [Modality.AUDIO],
          speechConfig: {
            voiceConfig: { prebuiltVoiceConfig: { voiceName: 'Kore' } },
          },
          systemInstruction,
          inputAudioTranscription: {},
          outputAudioTranscription: {},
        },
      });

      // Prevent screen sleep while actively recording
      requestWakeLock();

      const session = await sessionPromiseRef.current;
      sessionRef.current = session;

      if (!isMountedRef.current) {
        stopRecording({ skipStateReset: true });
        return;
      }

      const InputContextCtor = window.AudioContext || (window as any).webkitAudioContext;
      const inputAudioContext = new InputContextCtor({ sampleRate: 16000 });
      audioContextRef.current = inputAudioContext;
      await inputAudioContext.resume();

      mediaStreamRef.current = await navigator.mediaDevices.getUserMedia({
        audio: {
          channelCount: 1,
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
        },
      });

      mediaStreamSourceRef.current = inputAudioContext.createMediaStreamSource(
        mediaStreamRef.current,
      );

      scriptProcessorRef.current = inputAudioContext.createScriptProcessor(4096, 1, 1);
      scriptProcessorRef.current.onaudioprocess = (event) => {
        const inputData = event.inputBuffer.getChannelData(0);
        const pcm = new Int16Array(inputData.length);

        for (let i = 0; i < inputData.length; i += 1) {
          const sample = Math.max(-1, Math.min(1, inputData[i]));
          pcm[i] = sample < 0 ? sample * 0x8000 : sample * 0x7fff;
        }

        const pcmBlob: Blob = {
          data: encode(new Uint8Array(pcm.buffer)),
          mimeType: 'audio/pcm;rate=16000',
        };

        const liveSession = sessionRef.current;
        if (liveSession) {
          try {
            liveSession.sendRealtimeInput({ media: pcmBlob });
          } catch (sendError) {
            console.error('Failed to send realtime audio chunk', sendError);
          }
        }
      };

      mediaStreamSourceRef.current.connect(scriptProcessorRef.current);
      scriptProcessorRef.current.connect(inputAudioContext.destination);

      isRecordingRef.current = true;
      if (isMountedRef.current) {
        setIsRecording(true);
        setStatus('listening');
      }
    } catch (errorStarting) {
      console.error('Error starting recording:', errorStarting);
      if (isMountedRef.current) {
        setError('Failed to start microphone. Please check permissions and refresh.');
      }
      stopRecording();
    }
  }, [selectedPairIndex, stopRecording]);

  const statusLabel =
    status === 'listening' ? '正在听' : status === 'connecting' ? '正在连接…' : '麦克风已关闭';
  const statusHint =
    status === 'listening'
      ? '同声传译已就绪，请开始说话。'
      : status === 'connecting'
      ? '正在建立音频连接…'
      : `点击麦克风，进行从 ${selectedPair.source} 到 ${selectedPair.target} 的口译。`;

  const micButtonClass = [
    'mic-button',
    status === 'listening' ? 'recording' : '',
    status === 'connecting' ? 'connecting' : '',
  ]
    .filter(Boolean)
    .join(' ');

  return (
    <div className={`container ${isMaximized ? 'maximized' : ''}`}>
      <div className="title-bar">
        <button
          className="settings-button"
          onClick={() => setIsSettingsOpen(true)}
          aria-label="Open settings"
        >
          <svg
            xmlns="http://www.w3.org/2000/svg"
            width="28"
            height="28"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
          >
            <path d="M12.22 2h-.44a2 2 0 0 0-2 2v.18a2 2 0 0 1-1 1.73l-.43.25a2 2 0 0 1-2 0l-.15-.08a2 2 0 0 0-2.73.73l-.22.38a2 2 0 0 0 .73 2.73l.15.1a2 2 0 0 1 0 2l-.15.08a2 2 0 0 0-.73 2.73l.22.38a2 2 0 0 0 2.73.73l.15-.08a2 2 0 0 1 2 0l.43.25a2 2 0 0 1 1 1.73V20a2 2 0 0 0 2 2h.44a2 2 0 0 0 2-2v-.18a2 2 0 0 1 1-1.73l.43-.25a2 2 0 0 1 2 0l.15.08a2 2 0 0 0 2.73-.73l.22-.38a2 2 0 0 0-.73-2.73l-.15-.08a2 2 0 0 1 0-2l.15.08a2 2 0 0 0 .73-2.73l-.22-.38a2 2 0 0 0-2.73-.73l-.15.08a2 2 0 0 1-2 0l-.43-.25a2 2 0 0 1-1-1.73V4a2 2 0 0 0-2-2z" />
            <circle cx="12" cy="12" r="3" />
          </svg>
        </button>
        <div className="title-stack">
          <h1>同声传译</h1>
          <p className="title-subtitle">基于 Gemini 的实时口译</p>
        </div>
        <button
          className="maximize-button"
          onClick={() => setIsMaximized((prev) => !prev)}
          aria-label={isMaximized ? 'Restore layout' : 'Maximize layout'}
        >
          {isMaximized ? (
            <svg
              xmlns="http://www.w3.org/2000/svg"
              width="24"
              height="24"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
            >
              <polyline points="4 14 10 14 10 20" />
              <polyline points="20 10 14 10 14 4" />
              <line x1="14" y1="10" x2="21" y2="3" />
              <line x1="3" y1="21" x2="10" y2="14" />
            </svg>
          ) : (
            <svg
              xmlns="http://www.w3.org/2000/svg"
              width="24"
              height="24"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
            >
              <path d="M8 3H5a2 2 0 0 0-2 2v3m18 0V5a2 2 0 0 0-2-2h-3m0 18h3a2 2 0 0 0 2-2v-3M3 16v3a2 2 0 0 0 2 2h3" />
            </svg>
          )}
        </button>
      </div>

      <div className="status-bar" role="status" aria-live="polite">
        <div className="language-chip">
          <span className="language-label">{selectedPair.source}</span>
          <span className="language-separator" aria-hidden="true">
            →
          </span>
          <span className="language-label">{selectedPair.target}</span>
        </div>
        <div className={`status-indicator status-${status}`}>
          <span className="status-dot" />
          <span>{statusLabel}</span>
        </div>
      </div>

      <div className="transcript-container" ref={transcriptRef}>
        {transcript.length === 0 ? (
          <div className="placeholder">
            <div className="placeholder-visual">
              <svg
                xmlns="http://www.w3.org/2000/svg"
                width="64"
                height="64"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="1.2"
                strokeLinecap="round"
                strokeLinejoin="round"
                className="placeholder-icon"
              >
                <path d="M12 2a3 3 0 0 0-3 3v7a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3Z" />
                <path d="M19 10v2a7 7 0 0 1-14 0v-2" />
                <line x1="12" x2="12" y1="19" y2="22" />
              </svg>
            </div>
            <p className="placeholder-title">暂无翻译记录</p>
            <p className="placeholder-copy">
              {status === 'idle'
                ? `点击麦克风即可从 ${selectedPair.source} 到 ${selectedPair.target} 进行口译。`
                : '已准备就绪，开始说话即可在此看到翻译结果。'}
            </p>
          </div>
        ) : (
          transcript.map((entry, index) => (
            <article key={`${entry.createdAt}-${index}`} className="transcript-entry">
              <header className="entry-meta">
                <span className="entry-time">{formatTimestamp(entry.createdAt)}</span>
              </header>
              <div className="entry-body">
                <div className="bubble bubble-source">
                  <span className="bubble-label">{entry.sourceLang}</span>
                  <p>{entry.sourceText}</p>
                </div>
                <div className="bubble bubble-target">
                  <span className="bubble-label">{entry.targetLang}</span>
                  <p>{entry.translatedText}</p>
                </div>
              </div>
            </article>
          ))
        )}
      </div>

      {error && (
        <div className="error-message" role="alert">
          {error}
        </div>
      )}

      <div className="controls">
        <button
          className={micButtonClass}
          onClick={status === 'idle' ? startRecording : () => stopRecording()}
          aria-label={status === 'idle' ? '开始口译' : '停止口译'}
        >
          {status === 'idle' ? (
            <svg xmlns="http://www.w3.org/2000/svg" width="28" height="28" viewBox="0 0 24 24" fill="currentColor">
              <path d="M12 14c1.66 0 2.99-1.34 2.99-3L15 5c0-1.66-1.34-3-3-3S9 3.34 9 5v6c0 1.66 1.34 3 3 3zm5.3-3c0 3-2.54 5.1-5.3 5.1S6.7 14 6.7 11H5c0 3.41 2.72 6.23 6 6.72V21h2v-3.28c3.28-.49 6-3.31 6-6.72h-1.7z" />
            </svg>
          ) : (
            <svg xmlns="http://www.w3.org/2000/svg" width="28" height="28" viewBox="0 0 24 24" fill="currentColor">
              <rect x="6" y="6" width="12" height="12" rx="2" />
            </svg>
          )}
        </button>
        <div className="control-info">
          <span className="control-heading">{statusLabel}</span>
          <p className="control-helper">{statusHint}</p>
          <button
            className="secondary-button"
            onClick={clearTranscript}
            disabled={transcript.length === 0}
          >
            清空记录
          </button>
        </div>
      </div>

      {isSettingsOpen && (
        <div className="settings-modal-overlay" onClick={() => setIsSettingsOpen(false)}>
          <div className="settings-modal" onClick={(event) => event.stopPropagation()}>
            <h2>设置</h2>
            <div className="setting-item">
              <label htmlFor="language-pair">翻译语言</label>
              <select
                id="language-pair"
                value={selectedPairIndex}
                onChange={(event) => handlePairChange(Number.parseInt(event.target.value, 10))}
              >
                {LANGUAGE_PAIRS.map((pair, index) => (
                  <option key={pair.name} value={index}>
                    {pair.name}
                  </option>
                ))}
              </select>
            </div>
            <div className="setting-item">
              <label htmlFor="speech-pace">语速</label>
              <select
                id="speech-pace"
                value={pace}
                onChange={(e) => setPace((e.target.value as 'slow'|'normal'|'fast'))}
              >
                <option value="slow">慢</option>
                <option value="normal">正常</option>
                <option value="fast">快</option>
              </select>
            </div>
            <div className="setting-item">
              <button className="clear-button" onClick={clearTranscript}>
                Clear transcript
              </button>
            </div>
            <button className="close-settings" onClick={() => setIsSettingsOpen(false)}>
              完成
            </button>
          </div>
        </div>
      )}
    </div>
  );
};

const root = ReactDOM.createRoot(document.getElementById('root')!);
root.render(<LiveTranslatorApp />);
