import yaml from 'js-yaml';

export function saveToMemory(message, reply) {
  const memory = {
    timestamp: new Date().toISOString(),
    input: message,
    response: reply
  };

  const yamlText = yaml.dump(memory);
  console.log('🧠 Saving to memory:', yamlText);

  // Placeholder: In real setup, you'd write to file using backend or API.
}
