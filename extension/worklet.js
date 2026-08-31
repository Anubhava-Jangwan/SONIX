/*
 * AudioWorkletProcessor: hands each block of captured samples to the main
 * thread. Runs on the audio render thread, so it does no allocation beyond the
 * copy it posts and no format conversion — that happens in offscreen.js.
 */
class SonixCapture extends AudioWorkletProcessor {
  process(inputs) {
    const channel = inputs[0][0];
    if (channel) this.port.postMessage(new Float32Array(channel));
    return true;   // keep the node alive even during silence
  }
}
registerProcessor("sonix-capture", SonixCapture);
