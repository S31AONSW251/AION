import React from 'react';

const SettingsModal = ({ showSettings, setShowSettings, settings, setSettings, voices, speak, soulState, isSpeechSupported }) => {
    if (!showSettings) return null;

    return (
        <div className="settings-modal">
            <div className="settings-content">
                <div className="settings-header">
                    <h2>Settings</h2>
                    <button
                        className="close-button"
                        onClick={() => setShowSettings(false)}
                    >
                        &times;
                    </button>
                </div>

                <div className="settings-tabs">
                    <button className="settings-tab active">Voice</button>
                    <button className="settings-tab">Appearance</button>
                    <button className="settings-tab">Behavior</button>
                    <button className="settings-tab">Search</button>
                    <button className="settings-tab">Math</button>
                    <button className="settings-tab">Quantum</button>
                    <button className="settings-tab">Neural</button>
                    <button className="settings-tab">Creative</button>
                    <button className="settings-tab">Memory</button>
                    <button className="settings-tab">Goals</button>
                    <button className="settings-tab">Knowledge</button>
                </div>

                <div className="settings-grid">
                  {/* Voice Settings */}
                    <div className="settings-group">
                        <h3>Voice Settings</h3>
                        <div className="setting-item">
                            <label>Voice</label>
                            <select
                                value={settings.voiceName}
                                onChange={(e) => setSettings({ ...settings, voiceName: e.target.value })}
                            >
                                {voices.map(voice => (
                                    <option key={voice.name} value={voice.name}>
                                        {voice.name} ({voice.lang})
                                    </option>
                                ))}
                            </select>
                        </div>
                        <div className="setting-item">
                            <label>Language: {settings.language}</label>
                            <select
                                value={settings.language}
                                onChange={(e) => setSettings({ ...settings, language: e.target.value })}
                            >
                                <option value="en-US">English (US)</option>
                                <option value="en-GB">English (UK)</option>
                                <option value="es-ES">Spanish</option>
                                <option value="fr-FR">French</option>
                                <option value="de-DE">German</option>
                            </select>
                        </div>
                        <div className="setting-item">
                            <label>Pitch: {settings.pitch.toFixed(1)}</label>
                            <input
                                type="range" min="0.1" max="2" step="0.1"
                                value={settings.pitch}
                                onChange={(e) => setSettings({ ...settings, pitch: parseFloat(e.target.value) })}
                            />
                        </div>
                        <div className="setting-item">
                            <label>Rate: {settings.rate.toFixed(1)}</label>
                            <input
                                type="range" min="0.1" max="10" step="0.1"
                                value={settings.rate}
                                onChange={(e) => setSettings({ ...settings, rate: parseFloat(e.target.value) })}
                            />
                        </div>
                        <div className="setting-item">
                            <label>Volume: {settings.volume.toFixed(1)}</label>
                            <input
                                type="range" min="0" max="1" step="0.1"
                                value={settings.volume}
                                onChange={(e) => setSettings({ ...settings, volume: parseFloat(e.target.value) })}
                            />
                        </div>
                    </div>
                    {/* Appearance Settings */}
                    <div className="settings-group">
                        <h3>Appearance</h3>
                        <div className="setting-item">
                            <label>Theme</label>
                            <select
                                value={settings.theme}
                                onChange={(e) => setSettings({ ...settings, theme: e.target.value })}
                            >
                                <option value="dark">Dark</option>
                                <option value="light">Light</option>
                            </select>
                        </div>
                        <div className="setting-item toggle">
                            <label>Animations</label>
                            <label className="switch">
                                <input type="checkbox" checked={settings.animationEnabled} onChange={(e) => setSettings({ ...settings, animationEnabled: e.target.checked })} />
                                <span className="slider"></span>
                            </label>
                        </div>
                        <div className="setting-item toggle">
                            <label>Sound Effects</label>
                            <label className="switch">
                                <input type="checkbox" checked={settings.soundEffects} onChange={(e) => setSettings({ ...settings, soundEffects: e.target.checked })} />
                                <span className="slider"></span>
                            </label>
                        </div>
                        <div className="setting-item toggle">
                            <label>Energy Saver</label>
                            <label className="switch">
                                <input type="checkbox" checked={settings.energySaver} onChange={(e) => setSettings({ ...settings, energySaver: e.target.checked })} />
                                <span className="slider"></span>
                            </label>
                        </div>
                    </div>
                    {/* Behavior Settings */}
                    <div className="settings-group">
                        <h3>Behavior</h3>
                        <div className="setting-item">
                            <label>Welcome Message</label>
                            <input
                                type="text"
                                value={settings.welcomeMessage}
                                onChange={(e) => setSettings({ ...settings, welcomeMessage: e.target.value })}
                            />
                        </div>
                        <div className="setting-item toggle">
                            <label>Auto Speak Replies</label>
                            <label className="switch">
                                <input type="checkbox" checked={settings.autoSpeakReplies} onChange={(e) => setSettings({ ...settings, autoSpeakReplies: e.target.checked })} />
                                <span className="slider"></span>
                            </label>
                        </div>
                        <div className="setting-item toggle">
                            <label>Auto Listen</label>
                            <label className="switch">
                                <input type="checkbox" checked={settings.autoListen} onChange={(e) => setSettings({ ...settings, autoListen: e.target.checked })} />
                                <span className="slider"></span>
                            </label>
                        </div>
                        <div className="setting-item toggle">
                            <label>Affirmation Loop</label>
                            <label className="switch">
                                <input type="checkbox" checked={settings.affirmationLoop} onChange={(e) => setSettings({ ...settings, affirmationLoop: e.target.checked })} />
                                <span className="slider"></span>
                            </label>
                        </div>
                        <div className="setting-item">
                            <label>Personality Intensity: {settings.personalityIntensity}%</label>
                            <input
                                type="range" min="0" max="100" step="1"
                                value={settings.personalityIntensity}
                                onChange={(e) => setSettings({ ...settings, personalityIntensity: parseInt(e.target.value) })}
                            />
                        </div>
                        <div className="setting-item toggle">
                            <label>Enable Sentiment Analysis</label>
                            <label className="switch">
                                <input type="checkbox" checked={settings.enableSentimentAnalysis} onChange={(e) => setSettings({ ...settings, enableSentimentAnalysis: e.target.checked })} />
                                <span className="slider"></span>
                            </label>
                        </div>
                        <div className="setting-item toggle">
                            <label>Enable Self-Correction</label>
                            <label className="switch">
                                <input type="checkbox" checked={settings.enableSelfCorrection} onChange={(e) => setSettings({ ...settings, enableSelfCorrection: e.target.checked })} />
                                <span className="slider"></span>
                            </label>
                        </div>
                    </div>
                    {/* Search Settings */}
                    <div className="settings-group">
                        <h3>Search Settings</h3>
                        <div className="setting-item toggle">
                            <label>Enable Web Search</label>
                            <label className="switch">
                                <input type="checkbox" checked={settings.enableWebSearch} onChange={(e) => setSettings({ ...settings, enableWebSearch: e.target.checked })} />
                                <span className="slider"></span>
                            </label>
                        </div>
                        <div className="setting-item">
                            <label>Search Provider</label>
                            <select
                                value={settings.searchProvider}
                                onChange={(e) => setSettings({ ...settings, searchProvider: e.target.value })}
                            >
                                <option value="google">Google</option>
                                <option value="bing">Bing</option>
                                <option value="wolfram">Wolfram Alpha</option>
                            </select>
                        </div>
                        <div className="setting-item">
                            <label>Search Depth: {settings.searchDepth}</label>
                            <input
                                type="range" min="1" max="10" step="1"
                                value={settings.searchDepth}
                                onChange={(e) => setSettings({ ...settings, searchDepth: parseInt(e.target.value) })}
                            />
                        </div>
                        <div className="setting-item">
                            <label>Real Search API Endpoint (Optional)</label>
                            <input
                                type="text"
                                value={settings.realSearchApiEndpoint}
                                onChange={(e) => setSettings({ ...settings, realSearchApiEndpoint: e.target.value })}
                                placeholder="e.g., https://your-backend.com/api/search"
                            />
                        </div>
                    </div>
                    {/* Math Settings */}
                    <div className="settings-group">
                        <h3>Math Settings</h3>
                        <div className="setting-item toggle">
                            <label>Enable Math Solving</label>
                            <label className="switch">
                                <input type="checkbox" checked={settings.enableMathSolving} onChange={(e) => setSettings({ ...settings, enableMathSolving: e.target.checked })} />
                                <span className="slider"></span>
                            </label>
                        </div>
                        <div className="setting-item toggle">
                            <label>Show Math Steps</label>
                            <label className="switch">
                                <input type="checkbox" checked={settings.showMathSteps} onChange={(e) => setSettings({ ...settings, showMathSteps: e.target.checked })} />
                                <span className="slider"></span>
                            </label>
                        </div>
                        <div className="setting-item">
                            <label>Math Engine</label>
                            <select
                                value={settings.mathEngine}
                                onChange={(e) => setSettings({ ...settings, mathEngine: e.target.value })}
                            >
                                <option value="mathjs">Math.js</option>
                                <option value="native">Native</option>
                            </select>
                        </div>
                    </div>
                    {/* Quantum Settings */}
                    <div className="settings-group">
                        <h3>Quantum Settings</h3>
                        <div className="setting-item toggle">
                            <label>Enable Quantum Features</label>
                            <label className="switch">
                                <input type="checkbox" checked={settings.enableQuantum} onChange={(e) => setSettings({ ...settings, enableQuantum: e.target.checked })} />
                                <span className="slider"></span>
                            </label>
                        </div>
                        <div className="setting-item">
                            <label>Quantum Depth: {settings.quantumDepth} qubits</label>
                            <input
                                type="range" min="1" max="5" step="1"
                                value={settings.quantumDepth}
                                onChange={(e) => setSettings({ ...settings, quantumDepth: parseInt(e.target.value) })}
                                disabled={!settings.enableQuantum}
                            />
                        </div>
                    </div>
                    {/* Neural Settings */}
                    <div className="settings-group">
                        <h3>Neural Settings</h3>
                        <div className="setting-item toggle">
                            <label>Enable Neural Features</label>
                            <label className="switch">
                                <input type="checkbox" checked={settings.enableNeural} onChange={(e) => setSettings({ ...settings, enableNeural: e.target.checked })} />
                                <span className="slider"></span>
                            </label>
                        </div>
                        <div className="setting-item">
                            <label>Hidden Layers: {settings.neuralLayers}</label>
                            <input
                                type="range" min="1" max="5" step="1"
                                value={settings.neuralLayers}
                                onChange={(e) => setSettings({ ...settings, neuralLayers: parseInt(e.target.value) })}
                                disabled={!settings.enableNeural}
                            />
                        </div>
                    </div>
                    {/* Creative Settings */}
                    <div className="settings-group">
                        <h3>Creative Settings</h3>
                        <div className="setting-item toggle">
                            <label>Enable Creative Generation</label>
                            <label className="switch">
                                <input type="checkbox" checked={settings.enableCreativeGeneration} onChange={(e) => setSettings({ ...settings, enableCreativeGeneration: e.target.checked })} />
                                <span className="slider"></span>
                            </label>
                        </div>
                        <div className="setting-item toggle">
                            <label>Enable Image Generation</label>
                            <label className="switch">
                                <input type="checkbox" checked={settings.enableImageGeneration} onChange={(e) => setSettings({ ...settings, enableImageGeneration: e.target.checked })} />
                                <span className="slider"></span>
                            </label>
                        </div>
                    </div>
                    {/* Memory Settings */}
                    <div className="settings-group">
                        <h3>Memory Settings</h3>
                        <div className="setting-item toggle">
                            <label>Enable Long-Term Memory</label>
                            <label className="switch">
                                <input type="checkbox" checked={settings.enableLongTermMemory} onChange={(e) => setSettings({ ...settings, enableLongTermMemory: e.target.checked })} />
                                <span className="slider"></span>
                            </label>
                        </div>
                        <div className="setting-item toggle">
                            <label>Enable Self-Reflection</label>
                            <label className="switch">
                                <input type="checkbox" checked={settings.enableSelfReflection} onChange={(e) => setSettings({ ...settings, enableSelfReflection: e.target.checked })} />
                                <span className="slider"></span>
                            </label>
                        </div>
                        <div className="setting-item">
                            <label>Reflection Frequency (ms): {settings.reflectionFrequency}</label>
                            <input
                                type="range" min="60000" max="600000" step="30000"
                                value={settings.reflectionFrequency}
                                onChange={(e) => setSettings({ ...settings, reflectionFrequency: parseInt(e.target.value) })}
                                disabled={!settings.enableSelfReflection}
                            />
                        </div>
                    </div>
                    {/* Goal Tracking */}
                    <div className="settings-group">
                        <h3>Goal Tracking</h3>
                        <div className="setting-item toggle">
                            <label>Enable Goal Tracking</label>
                            <label className="switch">
                                <input type="checkbox" checked={settings.goalTracking} onChange={(e) => setSettings({ ...settings, goalTracking: e.target.checked })} />
                                <span className="slider"></span>
                            </label>
                        </div>
                    </div>
                    {/* Knowledge Base */}
                    <div className="settings-group">
                        <h3>Knowledge Base</h3>
                        <div className="setting-item toggle">
                            <label>Enable Knowledge Base</label>
                            <label className="switch">
                                <input type="checkbox" checked={settings.knowledgeBase} onChange={(e) => setSettings({ ...settings, knowledgeBase: e.target.checked })} />
                                <span className="slider"></span>
                            </label>
                        </div>
                    </div>
                </div>

                <div className="settings-footer">
                    <button
                        className="test-button"
                        onClick={() => speak("This is a voice test. My current mood is " + soulState.currentMood)}
                        disabled={!isSpeechSupported}
                    >
                        Test Voice
                    </button>
                    <button
                        className="save-button"
                        onClick={() => setShowSettings(false)}
                    >
                        Save Settings
                    </button>
                </div>
            </div>
        </div>
    );
};

export default SettingsModal;