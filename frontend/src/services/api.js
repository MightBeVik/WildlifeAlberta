/**
 * API service for communicating with the FastAPI backend.
 */
const API_BASE = '';  // Uses Vite proxy in dev

export const api = {
  /**
   * Upload a single image file.
   */
  async uploadImage(file) {
    const formData = new FormData();
    formData.append('file', file);
    const res = await fetch(`${API_BASE}/api/upload`, {
      method: 'POST',
      body: formData,
    });
    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || 'Upload failed');
    }
    return res.json();
  },

  /**
   * Upload multiple images.
   */
  async uploadBatch(files) {
    const formData = new FormData();
    files.forEach(f => formData.append('files', f));
    const res = await fetch(`${API_BASE}/api/upload/batch`, {
      method: 'POST',
      body: formData,
    });
    if (!res.ok) throw new Error('Batch upload failed');
    return res.json();
  },

  /**
   * List all uploaded images.
   */
  async getImages() {
    const res = await fetch(`${API_BASE}/api/images`);
    if (!res.ok) throw new Error('Failed to fetch images');
    return res.json();
  },

  /**
   * Run detection on an image.
   */
  async detectAnimals(imageId, confidence = 0.3) {
    const res = await fetch(
      `${API_BASE}/api/detect/${imageId}?confidence=${confidence}`,
      { method: 'POST' }
    );
    if (!res.ok) throw new Error('Detection failed');
    return res.json();
  },

  /**
   * Run species classification on an image (requires detection first).
   */
  async classifySpecies(imageId) {
    const res = await fetch(`${API_BASE}/api/classify/${imageId}`, {
      method: 'POST',
    });
    if (!res.ok) throw new Error('Classification failed');
    return res.json();
  },

  /**
   * Get all detection results.
   */
  async getDetections() {
    const res = await fetch(`${API_BASE}/api/detections`);
    if (!res.ok) throw new Error('Failed to fetch detections');
    return res.json();
  },

  /**
   * Get all classification results.
   */
  async getClassifications() {
    const res = await fetch(`${API_BASE}/api/classifications`);
    if (!res.ok) throw new Error('Failed to fetch classifications');
    return res.json();
  },

  /**
   * Get system stats.
   */
  async getStats() {
    const res = await fetch(`${API_BASE}/api/stats`);
    if (!res.ok) throw new Error('Failed to fetch stats');
    return res.json();
  },

  /**
   * Health check.
   */
  async healthCheck() {
    const res = await fetch(`${API_BASE}/api/health`);
    if (!res.ok) throw new Error('Health check failed');
    return res.json();
  },

  /**
   * Delete an image.
   */
  async deleteImage(imageId) {
    const res = await fetch(`${API_BASE}/api/images/${imageId}`, {
      method: 'DELETE',
    });
    if (!res.ok) throw new Error('Delete failed');
    return res.json();
  },

  /**
   * Full pipeline: upload → detect → classify
   */
  async processImage(file, confidence = 0.3) {
    const upload = await this.uploadImage(file);
    const detection = await this.detectAnimals(upload.image_id, confidence);
    let classification = null;
    if (detection.has_animal) {
      classification = await this.classifySpecies(upload.image_id);
    }
    return { upload, detection, classification };
  },
};
