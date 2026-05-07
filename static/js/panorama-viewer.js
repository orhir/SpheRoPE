/**
 * Panorama Viewer - Equirectangular projection viewer using Three.js
 * Adapted from the user study server viewer.
 */

const PANO_VERT = `varying vec2 vUv; void main(){vUv=uv;gl_Position=vec4(position,1.0);}`;
const PANO_FRAG = `
  #define PI 3.14159265359
  uniform sampler2D erpTexture;
  uniform float uYaw, uPitch, uFov, uAspect;
  varying vec2 vUv;
  void main(){
    vec2 ndc = (vUv - 0.5) * 2.0;
    float ht = tan(uFov / 2.0);
    vec3 ray = normalize(vec3(1.0, ndc.x * ht * uAspect, -ndc.y * ht));
    float cp = cos(uPitch), sp = sin(uPitch);
    mat3 Ry = mat3(cp,0,-sp, 0,1,0, sp,0,cp);
    float cy = cos(uYaw), sy = sin(uYaw);
    mat3 Rz = mat3(cy,-sy,0, sy,cy,0, 0,0,1);
    vec3 rd = Rz * Ry * ray;
    float phi = atan(rd.y, rd.x);
    float theta = asin(clamp(rd.z, -1.0, 1.0));
    gl_FragColor = texture2D(erpTexture, vec2(phi/(2.0*PI)+0.5, 0.5-theta/PI));
  }`;

class PanoramaViewer {
  constructor(containerEl, url, options = {}) {
    this.container = containerEl;
    this.lon = options.startLon || 0;
    this.lat = options.startLat || 0;
    this.fov = options.fov || 90;
    this.autoPan = options.autoPan !== undefined ? options.autoPan : true;
    this.autoPanSpeed = options.autoPanSpeed || 0.06;
    this.isDown = false;
    this.animTarget = null;
    this.animStart = null;
    this.animDur = 400;
    this.linkedViewers = [];
    this.isVideo = options.isVideo || false;
    this.videoEl = null;

    const w = containerEl.clientWidth;
    const h = containerEl.clientHeight;

    let tex;
    if (this.isVideo) {
      tex = this._createVideoTexture(url);
    } else {
      tex = this._createImageTexture(url);
    }

    this.material = new THREE.ShaderMaterial({
      uniforms: {
        erpTexture: { value: tex },
        uYaw: { value: 0 },
        uPitch: { value: 0 },
        uFov: { value: this.fov * Math.PI / 180 },
        uAspect: { value: w / h }
      },
      vertexShader: PANO_VERT,
      fragmentShader: PANO_FRAG,
      depthTest: false
    });

    this.scene = new THREE.Scene();
    this.camera = new THREE.Camera();
    this.scene.add(new THREE.Mesh(new THREE.PlaneGeometry(2, 2), this.material));

    this.renderer = new THREE.WebGLRenderer({ antialias: true });
    this.renderer.setPixelRatio(window.devicePixelRatio);
    this.renderer.setSize(w, h);
    containerEl.appendChild(this.renderer.domElement);

    this._bindEvents();
    this._animate();
  }

  _createImageTexture(url) {
    // Use a plain <img> without crossOrigin so that same-origin file:// loads
    // are not flagged as tainted by WebGL in strict browsers.
    const img = new Image();
    const tex = new THREE.Texture(img);
    tex.minFilter = THREE.LinearFilter;
    tex.magFilter = THREE.LinearFilter;
    img.onload = () => { tex.needsUpdate = true; };
    img.onerror = (e) => { console.warn('Panorama image failed to load:', url, e); };
    img.src = url;
    return tex;
  }

  _createVideoTexture(url) {
    const video = document.createElement('video');
    // Do NOT set crossOrigin when running from file://; otherwise the video
    // is treated as tainted and WebGL refuses to sample it.
    video.src = url;
    video.loop = true;
    video.muted = true;
    video.playsInline = true;
    video.play();
    this.videoEl = video;
    const tex = new THREE.VideoTexture(video);
    tex.minFilter = THREE.LinearFilter;
    tex.magFilter = THREE.LinearFilter;
    return tex;
  }

  linkWith(otherViewer) {
    if (!this.linkedViewers.includes(otherViewer)) {
      this.linkedViewers.push(otherViewer);
    }
    if (!otherViewer.linkedViewers.includes(this)) {
      otherViewer.linkedViewers.push(this);
    }
    // Ensure all linked viewers know about each other (group linking)
    this.linkedViewers.forEach(v => {
      otherViewer.linkedViewers.forEach(u => {
        if (v !== u && !v.linkedViewers.includes(u)) {
          v.linkedViewers.push(u);
          u.linkedViewers.push(v);
        }
      });
    });
  }

  navigateTo(lon, lat, animate = true) {
    this.autoPan = false;
    if (animate) {
      this.animStart = { lon: this.lon, lat: this.lat, time: performance.now() };
      this.animTarget = { lon, lat };
    } else {
      this.lon = lon;
      this.lat = lat;
    }
    this.linkedViewers.forEach(v => {
      v.autoPan = false;
      if (animate) {
        v.animStart = { lon: v.lon, lat: v.lat, time: performance.now() };
        v.animTarget = { lon, lat };
      } else {
        v.lon = lon;
        v.lat = lat;
      }
    });
  }

  resize() {
    const w = this.container.clientWidth;
    const h = this.container.clientHeight;
    this.renderer.setSize(w, h);
    this.material.uniforms.uAspect.value = w / h;
  }

  updateTexture(imageUrl) {
    const tex = this._createImageTexture(imageUrl);
    this.material.uniforms.erpTexture.value = tex;
  }

  _bindEvents() {
    const onDown = (e) => {
      this.isDown = true;
      this.autoPan = false;
      this.animTarget = null;
      this.linkedViewers.forEach(v => { v.autoPan = false; v.animTarget = null; });
      const p = e.touches ? e.touches[0] : e;
      this._startX = p.clientX;
      this._startY = p.clientY;
      this._lonStart = this.lon;
      this._latStart = this.lat;
      e.preventDefault();
    };

    const onMove = (e) => {
      if (!this.isDown) return;
      const p = e.touches ? e.touches[0] : e;
      this.lon = (this._startX - p.clientX) * 0.2 + this._lonStart;
      this.lat = (p.clientY - this._startY) * 0.2 + this._latStart;
      this.linkedViewers.forEach(v => { v.lon = this.lon; v.lat = this.lat; });
    };

    const onUp = () => { this.isDown = false; };

    this.container.addEventListener('mousedown', onDown);
    this.container.addEventListener('touchstart', onDown, { passive: false });
    document.addEventListener('mousemove', onMove);
    document.addEventListener('touchmove', onMove, { passive: false });
    document.addEventListener('mouseup', onUp);
    document.addEventListener('touchend', onUp);
  }

  _easeInOut(t) {
    return t < 0.5 ? 2 * t * t : -1 + (4 - 2 * t) * t;
  }

  _animate() {
    requestAnimationFrame(() => this._animate());

    if (this.autoPan) this.lon += this.autoPanSpeed;

    if (this.animTarget && this.animStart) {
      const t = Math.min((performance.now() - this.animStart.time) / this.animDur, 1);
      const e = this._easeInOut(t);
      this.lon = this.animStart.lon + (this.animTarget.lon - this.animStart.lon) * e;
      this.lat = this.animStart.lat + (this.animTarget.lat - this.animStart.lat) * e;
      if (t >= 1) this.animTarget = null;
    }

    this.lat = Math.max(-89, Math.min(89, this.lat));
    const yaw = this.lon * Math.PI / 180;
    const pitch = this.lat * Math.PI / 180;
    this.material.uniforms.uYaw.value = yaw;
    this.material.uniforms.uPitch.value = pitch;
    this.renderer.render(this.scene, this.camera);
  }
}
