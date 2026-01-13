window.onload=function(){


// HTML: <canvas id="glcanvas" width="300" height="300"></canvas>
const canvas = document.getElementById("glcanvas");
const gl = canvas.getContext("webgl");
if (!gl) alert("WebGL not supported");
// CSS: html, body {margin: 0; height: 100%; background: #111; display: flex; justify-content: center; align-items: center;}; canvas {background: #000; border: 1px solid #333;}
gl.enable(gl.BLEND);
gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
// Utility: Create audio context once for efficiency
const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
function playFieldEffect() {
const now = audioCtx.currentTime;
const duration = 1.3;
// Core oscillators
const carrier = audioCtx.createOscillator();
const modulator = audioCtx.createOscillator(); // FM
const lfo = audioCtx.createOscillator(); // vibrato
const modGain = audioCtx.createGain(); // FM depth
const lfoGain = audioCtx.createGain(); // vibrato depth
const gain = audioCtx.createGain();    // master volume
// Set base frequencies
const baseFreq = 60; //
carrier.type = "sawtooth";
modulator.type = "sine";
lfo.type = "sine";
// Slight FM modulation depth (creates “warble” texture)
modGain.gain.setValueAtTime(43, now); // Hz deviation for FM
lfoGain.gain.setValueAtTime(10, now); // Hz deviation for vibrato
// Connect modulator to carrier frequency (FM)
modulator.connect(modGain);
modGain.connect(carrier.frequency);
// Connect LFO to carrier frequency (vibrato)
lfo.connect(lfoGain);
lfoGain.connect(carrier.frequency);
// Envelope shaping
gain.gain.setValueAtTime(0.85, now);
gain.gain.exponentialRampToValueAtTime(0.3, now + 0.1); // attack
gain.gain.exponentialRampToValueAtTime(0.005, now + duration); // release
// Routing
carrier.connect(gain);
gain.connect(audioCtx.destination);
// Modulator / LFO params
modulator.frequency.setValueAtTime(baseFreq * 3, now); // 2x carrier
lfo.frequency.setValueAtTime(5, now); // slow vibrato
// Pitch bend upward
carrier.frequency.setValueAtTime(baseFreq, now);
carrier.frequency.linearRampToValueAtTime(baseFreq * 1.55, now + duration);
// Start and stop all
carrier.start(now);
modulator.start(now);
lfo.start(now);
carrier.stop(now + duration);
modulator.stop(now + duration);
lfo.stop(now + duration);
}
// -----------------------------------------------------------------------------
// Enhanced Rocket continuous engine sound
// -----------------------------------------------------------------------------
let rocketSound = null;
function startRocketHum() {
if (rocketSound) return; // already active
const now = audioCtx.currentTime;
// ======================================================
// 1️ Core oscillator - deep sub rumble
// ======================================================
const osc = audioCtx.createOscillator();
osc.type = "triangle";
osc.frequency.setValueAtTime(25, now);
// Filter the rumble a bit for realism
const humFilter = audioCtx.createBiquadFilter();
humFilter.type = "bandpass";
humFilter.frequency.setValueAtTime(90, now); // band center frequency (Hz)
humFilter.Q.setValueAtTime(0.9, now);
// ======================================================
// 2️ Generate reusable white noise buffer
// ======================================================
const bufferSize = audioCtx.sampleRate * 2;
const buffer = audioCtx.createBuffer(1, bufferSize, audioCtx.sampleRate);
const data = buffer.getChannelData(0);
for (let i = 0; i < bufferSize; i++) data[i] = Math.random() * 2 - 1;
// ======================================================
// 3️ Create three filtered noise variants
// ======================================================
const noiseHP = audioCtx.createBufferSource();
const noiseLP = audioCtx.createBufferSource();
const noiseBP = audioCtx.createBufferSource();
noiseHP.buffer = buffer;
noiseLP.buffer = buffer;
noiseBP.buffer = buffer;
noiseHP.loop = noiseLP.loop = noiseBP.loop = true;
// --- Filters ---
const highpassFilter = audioCtx.createBiquadFilter();
highpassFilter.type = "highpass";
highpassFilter.frequency.setValueAtTime(2500, now); //  HPF cutoff
highpassFilter.Q.setValueAtTime(0.7, now);
const lowpassFilter = audioCtx.createBiquadFilter();
lowpassFilter.type = "lowpass";
lowpassFilter.frequency.setValueAtTime(430, now); //  LPF cutoff
lowpassFilter.Q.setValueAtTime(0.8, now);
const bandpassFilter = audioCtx.createBiquadFilter();
bandpassFilter.type = "bandpass";
bandpassFilter.frequency.setValueAtTime(1000, now); //  BPF center freq
bandpassFilter.Q.setValueAtTime(1.0, now);
// ======================================================
// 4️ Gain controls (volume for each layer)
// ======================================================
const mainGain = audioCtx.createGain();   // oscillator
const hpGain = audioCtx.createGain();     // highpass noise
const lpGain = audioCtx.createGain();     // lowpass noise
const bpGain = audioCtx.createGain();     // bandpass noise
// --- Set relative levels ---
mainGain.gain.setValueAtTime(0.0001, now);
mainGain.gain.exponentialRampToValueAtTime(0.25, now + 0.2);
hpGain.gain.setValueAtTime(0.0001, now);
hpGain.gain.exponentialRampToValueAtTime(0.03, now + 0.2); // top hiss
lpGain.gain.setValueAtTime(0.0001, now);
lpGain.gain.exponentialRampToValueAtTime(0.08, now + 0.2); // low wind
bpGain.gain.setValueAtTime(0.0001, now);
bpGain.gain.exponentialRampToValueAtTime(0.05, now + 0.2); // mid whoosh
// ======================================================
// 5️ Connect signal chains
// ======================================================
osc.connect(humFilter).connect(mainGain);
noiseHP.connect(highpassFilter).connect(hpGain);
noiseLP.connect(lowpassFilter).connect(lpGain);
noiseBP.connect(bandpassFilter).connect(bpGain);
// Mix all to destination
mainGain.connect(audioCtx.destination);
hpGain.connect(audioCtx.destination);
lpGain.connect(audioCtx.destination);
bpGain.connect(audioCtx.destination);
// ======================================================
// 6️ Start playback
// ======================================================
osc.start(now);
noiseHP.start(now);
noiseLP.start(now);
noiseBP.start(now);
// Store for later stop
rocketSound = {
 osc, noiseHP, noiseLP, noiseBP,
 mainGain, hpGain, lpGain, bpGain
};
}
function stopRocketHum() {
if (!rocketSound) return;
const { osc, noiseHP, noiseLP, noiseBP, mainGain, hpGain, lpGain, bpGain } = rocketSound;
const now = audioCtx.currentTime;
// Fade all out
const fadeTime = 0.3;
[mainGain, hpGain, lpGain, bpGain].forEach(g => {
 g.gain.exponentialRampToValueAtTime(0.0001, now + fadeTime);
});
// Stop everything
osc.stop(now + fadeTime);
noiseHP.stop(now + fadeTime);
noiseLP.stop(now + fadeTime);
noiseBP.stop(now + fadeTime);
rocketSound = null;
}
function playChirp() {
const oscillator = audioCtx.createOscillator();
const gainNode = audioCtx.createGain();
oscillator.connect(gainNode);
gainNode.connect(audioCtx.destination);
const duration = 0.15;
const startFreq = 1800;
const endFreq = 50;
const now = audioCtx.currentTime;
oscillator.frequency.setValueAtTime(startFreq, now);
oscillator.frequency.linearRampToValueAtTime(endFreq, now + duration);
gainNode.gain.setValueAtTime(0.2, now);
gainNode.gain.exponentialRampToValueAtTime(0.001, now + duration);
oscillator.type = 'sine';
oscillator.start(now);
oscillator.stop(now + duration);
}
function playThud() {
const duration = 0.35;
const now = audioCtx.currentTime;
// --- Low chirp ---
const osc = audioCtx.createOscillator();
const gainOsc = audioCtx.createGain();
osc.connect(gainOsc);
gainOsc.connect(audioCtx.destination);
osc.type = 'sine';
osc.frequency.setValueAtTime(350, now);
osc.frequency.linearRampToValueAtTime(20, now + duration);
gainOsc.gain.setValueAtTime(0.9, now);
gainOsc.gain.exponentialRampToValueAtTime(0.02, now + duration);
// --- Noise burst ---
const bufferSize = audioCtx.sampleRate * duration;
const buffer = audioCtx.createBuffer(1, bufferSize, audioCtx.sampleRate);
const data = buffer.getChannelData(0);
for (let i = 0; i < bufferSize; i++) data[i] = Math.random() * 10 - 5;
const noise = audioCtx.createBufferSource();
noise.buffer = buffer;
const noiseGain = audioCtx.createGain();
noiseGain.gain.setValueAtTime(0.3, now);
noiseGain.gain.exponentialRampToValueAtTime(0.001, now + duration);
const filter = audioCtx.createBiquadFilter();
filter.type = 'lowpass';
filter.frequency.setValueAtTime(1100, now); // muffled “thud” feel
noise.connect(filter);
filter.connect(noiseGain);
noiseGain.connect(audioCtx.destination);
osc.start(now);
osc.stop(now + duration);
noise.start(now);
noise.stop(now + duration);}
function playReset() {
const notes = [329.63, 415.30, 493.88, 659.26]; // E4, G#4, B4, E5, E major arpeggio / triad thing
const duration = 0.3; // total
const noteDuration = duration / notes.length;
const now = audioCtx.currentTime;


notes.forEach((freq, i) => {
const osc = audioCtx.createOscillator();
const gain = audioCtx.createGain();
osc.connect(gain);
gain.connect(audioCtx.destination);


osc.type = 'triangle';
osc.frequency.setValueAtTime(freq, now + i * noteDuration);
gain.gain.setValueAtTime(0.2, now + i * noteDuration);
gain.gain.exponentialRampToValueAtTime(0.001, now + (i + 1) * noteDuration);


osc.start(now + i * noteDuration);
osc.stop(now + (i + 1) * noteDuration);
});
}
function playBounce() {
const oscillator = audioCtx.createOscillator();
const gainNode = audioCtx.createGain();
oscillator.connect(gainNode);
gainNode.connect(audioCtx.destination);
const duration = 0.2;
const startFreq = 120;
const endFreq = 400;
const now = audioCtx.currentTime;
oscillator.frequency.setValueAtTime(startFreq, now);
oscillator.frequency.linearRampToValueAtTime(endFreq, now + duration);
gainNode.gain.setValueAtTime(0.2, now);
gainNode.gain.exponentialRampToValueAtTime(0.001, now + duration);
oscillator.type = 'triangle';
oscillator.start(now);
oscillator.stop(now + duration);}
function playfuturebounce() {
const oscillator = audioCtx.createOscillator();
const gainNode = audioCtx.createGain();
oscillator.connect(gainNode);
gainNode.connect(audioCtx.destination);
const duration = 0.3;
const startFreq = 50;
const endFreq = 250;
const now = audioCtx.currentTime;
oscillator.frequency.setValueAtTime(startFreq, now);
oscillator.frequency.linearRampToValueAtTime(endFreq, now + duration);
gainNode.gain.setValueAtTime(0.2, now);
gainNode.gain.exponentialRampToValueAtTime(0.001, now + duration);
oscillator.type = 'triangle';
oscillator.start(now);
oscillator.stop(now + duration);}
function playfuturebounce2() {
const oscillator = audioCtx.createOscillator();
const gainNode = audioCtx.createGain();
oscillator.connect(gainNode);
gainNode.connect(audioCtx.destination);
const duration = 0.2;
const startFreq = 1000;
const endFreq = 900;
const now = audioCtx.currentTime;
oscillator.frequency.setValueAtTime(startFreq, now);
oscillator.frequency.linearRampToValueAtTime(endFreq, now + duration);
gainNode.gain.setValueAtTime(0.2, now);
gainNode.gain.exponentialRampToValueAtTime(0.001, now + duration);
oscillator.type = 'triangle';
oscillator.start(now);
oscillator.stop(now + duration);}
function playtick() {
const oscillator = audioCtx.createOscillator();
const gainNode = audioCtx.createGain();
oscillator.connect(gainNode);
gainNode.connect(audioCtx.destination);
const duration = 0.013;
const startFreq = 2000;
const endFreq = 1000;
const now = audioCtx.currentTime;
oscillator.frequency.setValueAtTime(startFreq, now);
oscillator.frequency.linearRampToValueAtTime(endFreq, now + duration);
gainNode.gain.setValueAtTime(0.2, now);
gainNode.gain.exponentialRampToValueAtTime(0.001, now + duration);
oscillator.type = 'triangle';
oscillator.start(now);
oscillator.stop(now + duration);}
function playtick2() {
const oscillator = audioCtx.createOscillator();
const gainNode = audioCtx.createGain();
oscillator.connect(gainNode);
gainNode.connect(audioCtx.destination);
const duration = 0.02;
const startFreq = 3000;
const endFreq = 1000;
const now = audioCtx.currentTime;
oscillator.frequency.setValueAtTime(startFreq, now);
oscillator.frequency.linearRampToValueAtTime(endFreq, now + duration);
gainNode.gain.setValueAtTime(0.2, now);
gainNode.gain.exponentialRampToValueAtTime(0.001, now + duration);
oscillator.type = 'sawtooth';
oscillator.start(now);
oscillator.stop(now + duration);}
// -----------------------------------------------------------------------------
// Utility Math
// -----------------------------------------------------------------------------
function sub(a,b){return[a[0]-b[0],a[1]-b[1],a[2]-b[2]];}
function cross(a,b){return[a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]];}
function normalize(v){const l=Math.hypot(v[0],v[1],v[2]);return l>0?[v[0]/l,v[1]/l,v[2]/l]:[0,0,0];}
function add(a,b){return[a[0]+b[0],a[1]+b[1],a[2]+b[2]];}
function mul(v,s){return[v[0]*s,v[1]*s,v[2]*s];}
function degToRad(d){return(d*Math.PI)/180;}
// -----------------------------------------------------------------------------
// Perspective & View
// -----------------------------------------------------------------------------
function perspective(fov,aspect,near,far){
const f=1.0/Math.tan(fov/2),nf=1/(near-far);
return [f/aspect,0,0,0,0,f,0,0,0,0,(far+near)*nf,-1,0,0,(2*far*near)*nf,0];}
function lookAt(eye,target,up){
const zx=eye[0]-target[0],zy=eye[1]-target[1],zz=eye[2]-target[2];
const zlen=Math.sqrt(zx*zx+zy*zy+zz*zz);
const z=[zx/zlen,zy/zlen,zz/zlen];
const x=[up[1]*z[2]-up[2]*z[1],up[2]*z[0]-up[0]*z[2],up[0]*z[1]-up[1]*z[0]];
const xlen=Math.sqrt(x[0]**2+x[1]**2+x[2]**2);
x[0]/=xlen;x[1]/=xlen;x[2]/=xlen;
const y=[z[1]*x[2]-z[2]*x[1],z[2]*x[0]-z[0]*x[2],z[0]*x[1]-z[1]*x[0]];
return [
x[0],y[0],z[0],0,
x[1],y[1],z[1],0,
x[2],y[2],z[2],0,
-(x[0]*eye[0]+x[1]*eye[1]+x[2]*eye[2]),
-(y[0]*eye[0]+y[1]*eye[1]+y[2]*eye[2]),
-(z[0]*eye[0]+z[1]*eye[1]+z[2]*eye[2]),1];}
function makeTranslationMatrix(tx,ty,tz){
return new Float32Array([
1,0,0,0,
0,1,0,0,
0,0,1,0,
tx,ty,tz,1]);}
// -----------------------------------------------------------------------------
// Quaternion Camera Controls
// -----------------------------------------------------------------------------
function quatNormalize(q){const l=Math.hypot(q[0],q[1],q[2],q[3]);return[q[0]/l,q[1]/l,q[2]/l,q[3]/l];}
function quatFromAxisAngle(axis,angle){const s=Math.sin(angle/2),c=Math.cos(angle/2);return[axis[0]*s,axis[1]*s,axis[2]*s,c];}
function quatMul(a,b){const[ax,ay,az,aw]=a,[bx,by,bz,bw]=b;return[aw*bx+ax*bw+ay*bz-az*by,aw*by-ax*bz+ay*bw+az*bx,aw*bz+ax*by-ay*bx+az*bw,aw*bw-ax*bx-ay*by-az*bz];}
function quatRotate(q,axis,angle){return quatNormalize(quatMul(quatFromAxisAngle(axis,angle),q));}
function quatToMatrix(q){const[x,y,z,w]=q;const xx=x*x,yy=y*y,zz=z*z;const xy=x*y,xz=x*z,yz=y*z;const wx=w*x,wy=w*y,wz=w*z;
return[1-2*(yy+zz),2*(xy+wz),2*(xz-wy),0,2*(xy-wz),1-2*(xx+zz),2*(yz+wx),0,2*(xz+wy),2*(yz-wx),1-2*(xx+yy),0,0,0,0,1];}
function vec2quat(dir){
let d=dir.slice();let len=Math.hypot(...d);
if(len<1e-8)return[0,0,0,1];
d=d.map(v=>v/len);
const f=[0,0,-1];
const cross=[f[1]*d[2]-f[2]*d[1],f[2]*d[0]-f[0]*d[2],f[0]*d[1]-f[1]*d[0]];
const dot=f[0]*d[0]+f[1]*d[1]+f[2]*d[2];
if(dot<-0.999999){
const axis=Math.abs(f[0])<0.1?[1,0,0]:[0,1,0];
const perp=[f[1]*axis[2]-f[2]*axis[1],f[2]*axis[0]-f[0]*axis[2],f[0]*axis[1]-f[1]*axis[0]];
const lenP=Math.hypot(...perp);
return[perp[0]/lenP,perp[1]/lenP,perp[2]/lenP,0];}
const w=Math.sqrt((1+dot)*2)/2;const scale=1/(2*w);
return[cross[0]*scale,cross[1]*scale,cross[2]*scale,w];}
// -----------------------------------------------------------------------------
// Surface Generation & Normals
// -----------------------------------------------------------------------------
function createSurface(xmin,xmax,nx,ymin,ymax,ny,f,g,h){
const vertices=[],normals=[],indices=[],cloud2image=[], image2cloud=[];
const dx=(xmax-xmin)/(nx-1),dy=(ymax-ymin)/(ny-1);
const points=[]
let counterthing=0;
for(let i=0;i<nx;i++){
const x=xmin+i*dx;points[i]=[];
for(let j=0;j<ny;j++){
const y=ymin+j*dy;
const vx=f(x,y),vy=g(x,y),vz=h(x,y);
vertices.push(vx,vy,vz);
cloud2image.push(i,j);
points[i][j]=[vx,vy,vz];
image2cloud.push(counterthing);
counterthing=counterthing+1;}}
for(let i=0;i<nx;i++){
for(let j=0;j<ny;j++){
const p=points[i][j];
const px=(i<nx-1)?points[i+1][j]:points[i-1][j];
const py=(j<ny-1)?points[i][j+1]:points[i][j-1];
let n;
if(i===nx-1&&j===ny-1)n=[0,0,0];
else if(i===nx-1||j===ny-1)n=normalize(cross(sub(py,p),sub(px,p)));
else n=normalize(cross(sub(px,p),sub(py,p)));
normals.push(...n);}}
for(let i=0;i<nx-1;i++){
for(let j=0;j<ny-1;j++){
const a=i*ny+j,b=(i+1)*ny+j;
indices.push(a,b,a+1,b,b+1,a+1);}}
//placeholder field=np.zeros((nx,ny,3))
const field=new Array(nx).fill().map(()=>new Array(ny).fill().map(()=>[0,0,0]));
return [vertices,normals,indices,cloud2image,image2cloud,field,nx,ny];}
function recomputeNormals(surface,nx,ny){
const [verts,norms]=surface;
const points=[];
for(let i=0;i<nx;i++){
points[i]=[];
for(let j=0;j<ny;j++){
const idx=i*ny+j;
points[i][j]=[verts[3*idx],verts[3*idx+1],verts[3*idx+2]];}}
const newNormals=[];
for(let i=0;i<nx;i++){
for(let j=0;j<ny;j++){
const p=points[i][j];
const px=(i<nx-1)?points[i+1][j]:points[i-1][j];
const py=(j<ny-1)?points[i][j+1]:points[i][j-1];
let n;
if(i===nx-1&&j===ny-1)n=[0,0,0];
else if(i===nx-1||j===ny-1)n=normalize(cross(sub(py,p),sub(px,p)));
else n=normalize(cross(sub(px,p),sub(py,p)));
newNormals.push(...n);}}
norms.splice(0,norms.length,...newNormals);}
// -----------------------------------------------------------------------------
// Shaders
// -----------------------------------------------------------------------------
const vs=`attribute vec3 aPosition;
attribute vec3 aNormal;
uniform mat4 uProjection,uView,uModel;
varying vec3 vNormal;
void main(){gl_Position=uProjection*uView*uModel*vec4(aPosition,1.0);vNormal=aNormal;}`;
const fs=`precision mediump float;
varying vec3 vNormal;
uniform vec3 uColor;uniform float uOpacity;
void main(){vec3 L=normalize(vec3(-1,-1,1));float d=dot(normalize(vNormal),L)*0.7+1.0;
gl_FragColor=vec4(uColor*d,uOpacity);}`;
function compileShader(src,type){const s=gl.createShader(type);gl.shaderSource(s,src);gl.compileShader(s);
return s;}
const prog=gl.createProgram();
gl.attachShader(prog,compileShader(vs,gl.VERTEX_SHADER));
gl.attachShader(prog,compileShader(fs,gl.FRAGMENT_SHADER));
gl.linkProgram(prog);gl.useProgram(prog);
const aPos=gl.getAttribLocation(prog,"aPosition");
const aNorm=gl.getAttribLocation(prog,"aNormal");
const uProj=gl.getUniformLocation(prog,"uProjection");
const uView=gl.getUniformLocation(prog,"uView");
const uModel=gl.getUniformLocation(prog,"uModel");
const uColor=gl.getUniformLocation(prog,"uColor");
const uOpacity=gl.getUniformLocation(prog,"uOpacity");
// -----------------------------------------------------------------------------
// Buffers + helper
// -----------------------------------------------------------------------------
function makeBuffers(o){
const[v,n,i]=o;
const vb=gl.createBuffer();gl.bindBuffer(gl.ARRAY_BUFFER,vb);gl.bufferData(gl.ARRAY_BUFFER,new Float32Array(v),gl.DYNAMIC_DRAW);
const nb=gl.createBuffer();gl.bindBuffer(gl.ARRAY_BUFFER,nb);gl.bufferData(gl.ARRAY_BUFFER,new Float32Array(n),gl.DYNAMIC_DRAW);
const ib=gl.createBuffer();gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER,ib);gl.bufferData(gl.ELEMENT_ARRAY_BUFFER,new Uint16Array(i),gl.STATIC_DRAW);
return {vb,nb,ib,count:i.length};}
function updateBuffers(){
sceneObjects.forEach(obj => {
const [v, n] = obj.geom;
const { vb, nb } = obj.buffers;
gl.bindBuffer(gl.ARRAY_BUFFER, vb);
gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(v), gl.DYNAMIC_DRAW);
gl.bindBuffer(gl.ARRAY_BUFFER, nb);
gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(n), gl.DYNAMIC_DRAW);});}
// -----------------------------------------------------------------------------
// Objects
// -----------------------------------------------------------------------------
const sphere=createSurface(0,2*Math.PI,80,0,Math.PI,50,
(phi,theta)=>3*Math.sin(theta)*Math.cos(phi),
(phi,theta)=>3*Math.sin(theta)*Math.sin(phi),
(phi,theta)=>3*Math.cos(theta));
// -----------------------------------------------------------------------------
// Scene Object Management
// -----------------------------------------------------------------------------
const sceneObjects = [];
function addObject(geom, color=[1,1,1], opacity=1, parent=null, offset=[0,0,0]) {
const buf = makeBuffers(geom);
sceneObjects.push({ geom, buffers: buf, color, opacity, parent, offset });
return sceneObjects.at(-1);}
// --- Main translucent wave sphere ---
const sphereObj = addObject(sphere, [0.2, 0.5, 1.0], 0.65);
let showWireframe = false;
function orientTowardOrigin() { // --- Ship ---
const dirToCenter = normalize(mul(shipPos, -1));
shipQuat = quatNormalize(vec2quat(dirToCenter));
// --- Camera ---
const dirCam = normalize(mul(camPos, -1));
camQuat = quatNormalize(vec2quat(dirCam));
console.log("Ship and camera oriented toward origin.");
// Force immediate redraw so seen instantly
updateBuffers();
updateCamera();
drawScene();}
// -----------------------------------------------------------------------------
// Spaceship State
// -----------------------------------------------------------------------------
let shipPos = [2.5, 2.5, 2.5];
let shipVel = [0, 0, 0];        // current velocity
let shipQuat = vec2quat([0, 0, 1]);    // orientation quaternion
const shipTurnSpeed = degToRad(1.0);
const shipThrust = 0.5;
// -----------------------------------------------------------------------------
// Spaceship Geometry (simple tetrahedron-ish)
// -----------------------------------------------------------------------------
const shipVerts = [
0, 0.03, 0,   // tip
-0.03, -0.03, 0,
0.03, -0.03, 0,
0, 0, -0.15];
const shipInds = [0,1,2, 0,2,3, 0,3,1, 1,2,3];
const shipNorms = new Array(shipVerts.length).fill(0); // flat shaded, fine for now
// Give the spaceship some scale so it's visible
for (let i = 0; i < shipVerts.length; i++) {shipVerts[i] *= 3.0;} // make it larger
// Give the ship fake normals so it's not all black
for (let i = 0; i < shipNorms.length; i += 3) {
shipNorms[i] = 0;
shipNorms[i + 1] = 0;
shipNorms[i + 2] = 1;}
const spaceship = [shipVerts, shipNorms, shipInds];
const shipObj = addObject(spaceship, [1.0, 0.2, 0.2], 1.0);
// -----------------------------------------------------------------------------
// Orange Flame (permanently attached below spaceship)
// -----------------------------------------------------------------------------
const flameBase = [
-0.04, -0.04, 0,
0.04, -0.04, 0,
0,    0.04,  0];
const flameTipHeightIdle = 0.0;
const flameTipHeightActive = 0.5;
function makeFlameVertices(height) {
return [
0, 0, height, // tip (height animates)
...flameBase];}
const flameVerts = makeFlameVertices(flameTipHeightIdle);
const flameInds = [0,1,2, 0,2,3, 0,3,1, 1,2,3];
const flameNorms = new Array(flameVerts.length).fill(0);
for (let i=0;i<flameNorms.length;i+=3) flameNorms[i+2]=1;


const flame = [flameVerts, flameNorms, flameInds];
const flameObj = addObject(flame, [1.0, 0.5, 0.0], 1.0, shipObj, [0, 0, 0.02]);
let flameCurrentHeight = flameTipHeightIdle;


const fov=degToRad(45),aspect=canvas.width/canvas.height;
gl.uniformMatrix4fv(uProj,false,new Float32Array(perspective(fov,aspect,0.1,100)));
let camQuat=vec2quat([-1,-1,-1]),camPos=[5,5,5];
const speed=0.5,turn=degToRad(2);
function updateCamera(){
if (cameraMounted) {
// placeholder: temporarily freeze camera when mounted
// (later this will track ship position/orientation)
const rot = quatToMatrix(shipQuat);
const shipForward = normalize([-rot[8], -rot[9], -rot[10]]);
const shipUp = normalize([rot[4], rot[5], rot[6]]);
const shipRight = normalize([rot[0], rot[1], rot[2]]);


// camera 2.5 units behind, 0.8 units above
const backOffset = mul(shipForward, -3);
const upOffset   = mul(shipUp, 1.5);
const camPosShip = add(add(shipPos, backOffset), upOffset);


const target = add(shipPos, add(mul(shipForward, 2.0), mul(shipUp, 0.3)));


gl.uniformMatrix4fv(uView, false, new Float32Array(lookAt(camPosShip, target, shipUp)));
return;}
const rot=quatToMatrix(camQuat);
const fwd=normalize([-rot[8],-rot[9],-rot[10]]);
const right=normalize([rot[0],rot[1],rot[2]]);
const up=normalize([rot[4],rot[5],rot[6]]);
if(keys['7'])camQuat=quatRotate(camQuat,right,+turn);
if(keys['8'])camQuat=quatRotate(camQuat,right,-turn);
if(keys['y'])camQuat=quatRotate(camQuat,up,+turn);
if(keys['i'])camQuat=quatRotate(camQuat,up,-turn);
if(keys['z'])camQuat=quatRotate(camQuat,fwd,-turn);
if(keys['c'])camQuat=quatRotate(camQuat,fwd,+turn);
if(keys['w'])camPos=add(camPos,mul(fwd,speed));
if(keys['s'])camPos=add(camPos,mul(fwd,-speed));
if(keys['a'])camPos=add(camPos,mul(right,-speed));
if(keys['d'])camPos=add(camPos,mul(right,+speed));
if(keys['x'])camPos=add(camPos,mul(up,speed));
if(keys['shift'])camPos=add(camPos,mul(up,-speed));
// -----------------------------------------------------------------------------
// Bounding box constraint for free-floating camera
// -----------------------------------------------------------------------------
for (let i = 0; i < 3; i++) {
if (camPos[i] > boundLimit) camPos[i] = boundLimit;
if (camPos[i] < -boundLimit) camPos[i] = -boundLimit;}
const target=add(camPos,fwd);
gl.uniformMatrix4fv(uView,false,new Float32Array(lookAt(camPos,target,up)));}
function updateSpaceship(dt) {
// Safety: ensure firing laser doesn't alter ship position
if (!keys['b'] && !gravityOn) {
shipVel = mul(shipVel, 0.999); // slight damping when idle
}
// ----------------------------------------------------
// rotation controls: j/l yaw, i/k pitch, m/. roll
// ----------------------------------------------------
// extract local axes from current orientation
const m = quatToMatrix(shipQuat);
const right = normalize([m[0], m[1], m[2]]);
const up    = normalize([m[4], m[5], m[6]]);
const forward = normalize([-m[8], -m[9], -m[10]]);
// yaw (left/right)
if (keys['h']) shipQuat = quatRotate(shipQuat, up, +shipTurnSpeed);
if (keys['k']) shipQuat = quatRotate(shipQuat, up, -shipTurnSpeed);
// pitch (up/down)
if (keys['u']) shipQuat = quatRotate(shipQuat, right, +shipTurnSpeed);
if (keys['j']) shipQuat = quatRotate(shipQuat, right, -shipTurnSpeed);
// roll (m = roll left, . = roll right)
if (keys['n']) shipQuat = quatRotate(shipQuat, forward, -shipTurnSpeed);
if (keys[',']) shipQuat = quatRotate(shipQuat, forward, +shipTurnSpeed);
if (keys['b']) {const accel = mul(forward, shipThrust);
shipVel = add(shipVel, mul(accel, dt * 60));} // scale to feel consistent
if (keys ['b'] || keys['n'] || keys[','] || keys['j'] || keys['u'] || keys['k'] || keys['h']) { startRocketHum(); // start looping engine sound
} else {stopRocketHum();} // stop when key released
// -----------------------------------------------------------------------------
// Apply gravity (if enabled)
// -----------------------------------------------------------------------------
if (gravityOn) {
const dirToCenter = normalize(mul(shipPos, -1));
const dist = Math.hypot(...shipPos);
const gForce = gravityStrength * dist * dt * 60 * 0.1;  // scaled for frame time
shipVel = add(shipVel, mul(dirToCenter, gForce));}
if (shipHoopCooldown <= 0 && checkHoopCollision(shipPos)) {
shipVel = mul(shipVel, -1);
playBounce()
shipHoopCooldown = HOOP_COOLDOWN_TIME;}
if (shipbackboardcooldown <= 0 && checkbackboardcollision(shipPos)) {
shipVel = mul(shipVel, -1);
playBounce()
shipbackboardcooldown = backboardcooldowntime;}
// motion integration
shipPos = add(shipPos, mul(shipVel, dt));
// -----------------------------------------------------------------------------
// Bounding box constraint for spaceship
// -----------------------------------------------------------------------------
for (let i = 0; i < 3; i++) {
if (shipPos[i] > boundLimit) {
shipPos[i] = boundLimit;
if (shipVel[i] > 0) { shipVel[i] *= -0.5; // bounce damping
if (shipboundbouncecooldown < 0)  {playBounce(); shipboundbouncecooldown=0.3}}
} else if (shipPos[i] < -boundLimit) {
shipPos[i] = -boundLimit;
if (shipVel[i] < 0) shipVel[i] *= -0.5;}}
// -----------------------------------------------------------------------------
// NEW: Green sphere collision projection
// -----------------------------------------------------------------------------
if (greenSphereOn) {
const distToCenter = Math.hypot(...shipPos);
if (distToCenter < greenRadius) {
// project ship back to the surface
const dir = normalize(shipPos);
shipPos = mul(dir, greenRadius);
// stop inward velocity component
const inward = (shipVel[0]*dir[0] + shipVel[1]*dir[1] + shipVel[2]*dir[2]);
if (inward < 0) {if (shiprefreshready) {playThud(); shiprefreshready=false}
shipVel = sub(shipVel, mul(dir, inward));}}
if (distToCenter > 3.5) {shiprefreshready=true}}
shipVel = mul(shipVel, 0.98); // simple drag
// -------------------------------------------------------------
// Detect movement — switch flame geometry
// Animate flame height smoothly
// -------------------------------------------------------------
const moving = keys['b'] || keys['u'] || keys['j'] || keys['h'] || keys['k'] || keys['n'] || keys[','];
const targetHeight = moving ? flameTipHeightActive : flameTipHeightIdle;
// smooth interpolation for flame height
flameCurrentHeight = flameCurrentHeight ?? flameTipHeightIdle;
flameCurrentHeight = flameCurrentHeight * 0.8 + targetHeight * 0.2;
const [fv, fn, fi] = flame;
const newVerts = makeFlameVertices(flameCurrentHeight);
fv.splice(0, fv.length, ...newVerts);
updateBuffers();}


function recomputeFlatNormals(verts, inds, norms) {
norms.fill(0);
for (let i = 0; i < inds.length; i += 3) {
const a = inds[i] * 3, b = inds[i + 1] * 3, c = inds[i + 2] * 3;
const vA = [verts[a], verts[a + 1], verts[a + 2]];
const vB = [verts[b], verts[b + 1], verts[b + 2]];
const vC = [verts[c], verts[c + 1], verts[c + 2]];
const n = normalize(cross(sub(vB, vA), sub(vC, vA)));
// accumulate
for (const v of [a, b, c]) {
norms[v]     += n[0];
norms[v + 1] += n[1];
norms[v + 2] += n[2];}}
// normalize each vertex normal
for (let i = 0; i < norms.length; i += 3) {
const n = normalize([norms[i], norms[i + 1], norms[i + 2]]);
norms[i] = n[0]; norms[i + 1] = n[1]; norms[i + 2] = n[2];}}
// -----------------------------------------------------------------------------
// Render (with white edge overlay)
// -----------------------------------------------------------------------------
function drawScene(){
gl.clearColor(0,0,0,1);
gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
gl.enable(gl.DEPTH_TEST);
// Sort: opaque first, transparent last
// Compute camera-space z for each object
function computeDepth(obj, viewMatrix) {
let pos;
if (obj === shipObj) pos = shipPos;
else if (obj === asteroidObj) pos = asteroidPos;
else if (obj.parent === shipObj) pos = add(shipPos, obj.offset);
else pos = [0,0,0];
return pos[0]*viewMatrix[2]
+ pos[1]*viewMatrix[6]
+ pos[2]*viewMatrix[10]
+ viewMatrix[14];}
// sort: opaque first, then translucent back-to-front
const opaque = [];
const translucent = [];
for (const obj of sceneObjects) {
if (obj.opacity >= 1.0) opaque.push(obj);
else translucent.push(obj);}
// compute depths for translucent
const viewMat = cameraMounted
? lookAt(camPos, add(camPos, [0,0,-1]), [0,1,0]): lookAt(camPos, add(camPos, [0,0,-1]), [0,1,0]);
translucent.sort((a, b) => computeDepth(b, viewMat) - computeDepth(a, viewMat));
const drawOrder = opaque.concat(translucent);
for (const obj of drawOrder){
const [v, n, i] = obj.geom;
const { vb, nb, ib, count } = obj.buffers;
gl.bindBuffer(gl.ARRAY_BUFFER, vb);
gl.vertexAttribPointer(aPos, 3, gl.FLOAT, false, 0, 0);
gl.enableVertexAttribArray(aPos);
gl.bindBuffer(gl.ARRAY_BUFFER, nb);
gl.vertexAttribPointer(aNorm, 3, gl.FLOAT, false, 0, 0);
gl.enableVertexAttribArray(aNorm);
gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, ib);
gl.uniform3fv(uColor, new Float32Array(obj.color));
gl.uniform1f(uOpacity, obj.opacity);
// --- Compute model matrix ---
let model = makeTranslationMatrix(0,0,0);
if (obj === sunCoreObj || obj === sunHaloObj) {
const t = obj.position;
model = makeTranslationMatrix(t[0], t[1], t[2]);
}
if (obj === hoopObj || obj === backboardObj || obj === netObj || obj === mobiusObj) {
const t = obj.position || [0,0,0];
model = makeTranslationMatrix(t[0], t[1], t[2]);
}


if (obj === shipObj) {
const rot = quatToMatrix(shipQuat);
const t = shipPos;
model = new Float32Array([
rot[0],rot[1],rot[2],0,
rot[4],rot[5],rot[6],0,
rot[8],rot[9],rot[10],0,
t[0],t[1],t[2],1
]);
}
else if (obj === asteroidObj) {
const s = asteroidVisualScale; // visual size factor
model = new Float32Array([
s, 0, 0, 0,
0, s, 0, 0,
0, 0, s, 0,
asteroidPos[0], asteroidPos[1], asteroidPos[2], 1
]);
}
else if (obj.parent === shipObj) {
const rot = quatToMatrix(shipQuat);
const t = shipPos;
const o = obj.offset;
const worldOffset = add(
add(mul([rot[0],rot[1],rot[2]],o[0]),
 mul([rot[4],rot[5],rot[6]],o[1])),
add(mul([rot[8],rot[9],rot[10]],o[2]),t)
);
model = new Float32Array([
rot[0],rot[1],rot[2],0,
rot[4],rot[5],rot[6],0,
rot[8],rot[9],rot[10],0,
worldOffset[0],worldOffset[1],worldOffset[2],1
]);
} else if (obj === laserObj && laserActive) {
const rot = quatToMatrix(laserQuat);
const t = laserPos;
model = new Float32Array([
rot[0],rot[1],rot[2],0,
rot[4],rot[5],rot[6],0,
rot[8],rot[9],rot[10],0,
t[0],t[1],t[2],1
]);
} else if (obj === frag1Obj) {
model = makeTranslationMatrix(frag1Pos[0], frag1Pos[1], frag1Pos[2]);
}
else if (obj === frag2Obj) {
model = makeTranslationMatrix(frag2Pos[0], frag2Pos[1], frag2Pos[2]);
}
gl.uniformMatrix4fv(uModel, false, model);
gl.depthMask(obj.opacity >= 1.0);
gl.drawElements(gl.TRIANGLES, count, gl.UNSIGNED_SHORT, 0);
// optional white wireframe overlay
if (showWireframe) {
// Push filled surface slightly back to avoid z-fighting
gl.enable(gl.POLYGON_OFFSET_FILL);
gl.polygonOffset(1.0, 1.0);
// Keep depth testing ON, but don't write depth for the lines
//gl.depthMask(false);
gl.uniform3fv(uColor, new Float32Array([1, 1, 1]));
gl.uniform1f(uOpacity, 1);
// Re-use same indices but draw edges
gl.drawElements(gl.LINES, count, gl.UNSIGNED_SHORT, 0);
// Restore defaults
gl.depthMask(true);
gl.disable(gl.POLYGON_OFFSET_FILL);
}}
gl.depthMask(true);}
let cameraMounted = false;
// -----------------------------------------------------------------------------
// NEW: Gravity + bounds + orient toggles
// -----------------------------------------------------------------------------
let gravityOn = false;          // toggled by 'g'
const gravityStrength = 0.05;   // adjust as desired
// -----------------------------------------------------------------------------
// Bounding box visualization (\pm boundLimit planes)
// -----------------------------------------------------------------------------
function makeBoundingPlane(axis, sign, size, color = [0.8,0.8,0.8], opacity = 0.05) {
const s = size;
let geom;
if (axis === 'x') {
geom = createSurface(-s, s, 2, -s, s, 2,
(y,z)=> sign * s, (y,z)=> y, (y,z)=> z);
} else if (axis === 'y') {
geom = createSurface(-s, s, 2, -s, s, 2,
(x,z)=> x, (x,z)=> sign * s, (x,z)=> z);
} else { // z-plane
geom = createSurface(-s, s, 2, -s, s, 2,
(x,y)=> x, (x,y)=> y, (x,y)=> sign * s);
}
return addObject(geom, color, opacity);
}
// create 6 planes forming a cube shell at ±boundLimit
const boundLimit = 20.0;
makeBoundingPlane('x', +1, boundLimit);
makeBoundingPlane('x', -1, boundLimit);
makeBoundingPlane('y', +1, boundLimit);
makeBoundingPlane('y', -1, boundLimit);
makeBoundingPlane('z', +1, boundLimit);
makeBoundingPlane('z', -1, boundLimit);
// -----------------------------------------------------------------------------
// Uniform spherical field of randomly oriented tetrahedrons starfield
// -----------------------------------------------------------------------------
function makestarfield(count, rInner = 15, rOuter = 25, seed = 42) {
const verts = [];
const inds = [];
const norms = [];
// --- Simple seeded RNG for repeatability ---
let s = seed >>> 0;
function rand() {
  s ^= s << 13; s ^= s >>> 17; s ^= s << 5;
  return (s >>> 0) / 4294967296;
}
// --- Helper for random unit quaternion rotation ---
function randomQuat() {
  const u1 = rand(), u2 = rand(), u3 = rand();
  const sq1 = Math.sqrt(1 - u1), sq2 = Math.sqrt(u1);
  return [
    Math.sin(2 * Math.PI * u2) * sq1,
    Math.cos(2 * Math.PI * u2) * sq1,
    Math.sin(2 * Math.PI * u3) * sq2,
    Math.cos(2 * Math.PI * u3) * sq2
  ];
}
// --- Rotate a vector by a quaternion ---
function rotateVec(v, q) {
  const [x, y, z] = v;
  const [qx, qy, qz, qw] = q;
  const uv = cross([qx, qy, qz], v);
  const uuv = cross([qx, qy, qz], uv);
  return add(v, add(mul(uv, 2 * qw), mul(uuv, 2)));
}
// --- Uniform spherical sampling ---
for (let t = 0; t < count; t++) {
  const u = rand();
  const v = rand();
  const theta = 2 * Math.PI * u;
  const phi = Math.acos(2 * v - 1);
  const dir = [
    Math.sin(phi) * Math.cos(theta),
    Math.sin(phi) * Math.sin(theta),
    Math.cos(phi)
  ];
  const xi = rand();
  const r = Math.cbrt(xi * (rOuter**3 - rInner**3) + rInner**3);
  const center = mul(dir, r);
  // base tetrahedron geometry (unit shape centered roughly near origin)
  const baseTetra = [
    [0, 0, 0],
    [1, 1, 0],
    [1, 0, 1],
    [0, 1, 1]
  ];
  // random orientation + uniform scale
  const q = randomQuat();
  const size = 0.2;
  const v0 = add(center, mul(rotateVec(baseTetra[0], q), size));
  const v1 = add(center, mul(rotateVec(baseTetra[1], q), size));
  const v2 = add(center, mul(rotateVec(baseTetra[2], q), size));
  const v3 = add(center, mul(rotateVec(baseTetra[3], q), size));
  const base = verts.length / 3;
  verts.push(...v0, ...v1, ...v2, ...v3);
  // 4 triangular faces of tetrahedron
  inds.push(base, base+1, base+2,
            base, base+2, base+3,
            base, base+3, base+1,
            base+1, base+3, base+2);
}
// compute vertex normals
for (let i = 0; i < verts.length; i++) norms.push(0);
recomputeFlatNormals(verts, inds, norms);
const geom = [verts, norms, inds];
addObject(geom, [1, 1, 1], 0.8);  // white stars, opaque
}
makestarfield(3500, 15, 80, 12345);
// -----------------------------------------------------------------------------
// Procedural spiral galaxy (parameterized position, orientation, size)
// -----------------------------------------------------------------------------
function makeSpiralGalaxy(center = [0,0,0], normal = [0,1,0], radius = 20, count = 3000, seed = 42) {
 const verts = [];
 const inds = [];
 const norms = [];
 // --- Seeded RNG for repeatability ---
 let s = seed >>> 0;
 function rand() {
   s ^= s << 13; s ^= s >>> 17; s ^= s << 5;
   return (s >>> 0) / 4294967296;
 }
 // --- Helper quaternion utilities (for random rotation + orientation) ---
 function randomQuat() {
   const u1 = rand(), u2 = rand(), u3 = rand();
   const sq1 = Math.sqrt(1 - u1), sq2 = Math.sqrt(u1);
   return [
     Math.sin(2 * Math.PI * u2) * sq1,
     Math.cos(2 * Math.PI * u2) * sq1,
     Math.sin(2 * Math.PI * u3) * sq2,
     Math.cos(2 * Math.PI * u3) * sq2
   ];}
 function rotateVec(v, q) {
   const [x, y, z] = v;
   const [qx, qy, qz, qw] = q;
   const uv = cross([qx, qy, qz], v);
   const uuv = cross([qx, qy, qz], uv);
   return add(v, add(mul(uv, 2 * qw), mul(uuv, 2)));}
 // --- Align galaxy plane normal to desired orientation ---
 const up = [0, 1, 0];
 const axis = normalize(cross(up, normal));
 const angle = Math.acos(Math.max(-1, Math.min(1, dot(up, normalize(normal)))));
 const sinHalf = Math.sin(angle / 2);
 const orientQ = [axis[0]*sinHalf, axis[1]*sinHalf, axis[2]*sinHalf, Math.cos(angle/2)];
 // --- Base tetrahedron shape ---
 const baseTetra = [
   [0, 0, 0],
   [1, 1, 0],
   [1, 0, 1],
   [0, 1, 1]
 ];
 // --- Spiral arm parameters ---
 const numArms = 11;
 const spiralTwist = 1.1 * Math.PI;  // total rotation of arms (controls number of windings)
 const armSpread = 0.04;           // angular fuzziness per arm
 const zSpread = 1;              // vertical thickness
 for (let t = 0; t < count; t++) { // which arm this star belongs to
   const arm = Math.floor(rand() * numArms); // radial distance biased toward center
   const u = rand(); const r = radius * Math.sqrt(u); // denser near center
   // base angle along spiral
   const theta = arm * (2 * Math.PI / numArms) + spiralTwist * (r / radius);
   // add some angular + vertical randomness
   const dTheta = (rand() - 0.5) * armSpread * 2 * Math.PI;
   const dz = (rand() - 0.5) * zSpread * radius * 0.1;
   // point in galaxy plane before orientation
   const localPos = [r * Math.cos(theta + dTheta),dz,r * Math.sin(theta + dTheta)];
   // rotate galaxy plane to desired orientation
   const worldPos = add(center, rotateVec(localPos, orientQ));
   // random tetrahedron orientation
   const q = randomQuat();
   const size = 0.1 + 0.1 * rand(); // small variation
   const v0 = add(worldPos, mul(rotateVec(baseTetra[0], q), size));
   const v1 = add(worldPos, mul(rotateVec(baseTetra[1], q), size));
   const v2 = add(worldPos, mul(rotateVec(baseTetra[2], q), size));
   const v3 = add(worldPos, mul(rotateVec(baseTetra[3], q), size));
   const base = verts.length / 3;
   verts.push(...v0, ...v1, ...v2, ...v3);
   // 4 triangular faces of tetrahedron
   inds.push(base, base+1, base+2,
             base, base+2, base+3,
             base, base+3, base+1,
             base+1, base+3, base+2);
 }
 // compute normals for lighting
 for (let i = 0; i < verts.length; i++) norms.push(0);
 recomputeFlatNormals(verts, inds, norms);
 const geom = [verts, norms, inds];
 addObject(geom, [1.0, 0.7, 0.5], 0.6);}
makeSpiralGalaxy([0, 0, 0], [0, 1, 0], 17, 1000, 999);
// -----------------------------------------------------------------------------
// Laser (purple pyramid)
// -----------------------------------------------------------------------------
const laserVerts = [
0, 0, 0.2,
-0.05,-0.05,0,
0.05,-0.05,0,
0, 0.05, 0
];
const laserInds = [0,1,2, 0,2,3, 0,3,1, 1,2,3];
const laserNorms = new Array(laserVerts.length).fill(0);
for (let i=0;i<laserNorms.length;i+=3) laserNorms[i+2]=1;
const laserGeom = [laserVerts, laserNorms, laserInds];
const laserObj = addObject(laserGeom, [0.8,0.2,1.0], 1.0);
laserObj.opacity = 0.0; // hidden initially
let laserActive = false;
let laserPos = [0,0,0];
let laserVel = [0,0,0];
let laserLife = 0;
let laserQuat = [0,0,0,1];
// -----------------------------------------------------------------------------
// Brown asteroid sphere basketball thing
// -----------------------------------------------------------------------------
let asteroidPos = [3,0,2];  //[3,0,2] center of mass
let asteroidVel = [0, 0, 0];
const asteroidRadius = 0.7;
const asteroidDrag = 0.999;
// --- Asteroid visual size states ---
let asteroidSizeLevels = [0.7, 0.55, 0.45, 0.3];
let asteroidSizeIndex = 0;  // starts at 0
let asteroidVisualScale = asteroidSizeLevels[asteroidSizeIndex];
function handleAsteroidHit() {
// advance to next size level
asteroidSizeIndex++;
// if we've gone past the smallest, reset
if (asteroidSizeIndex >= asteroidSizeLevels.length) {asteroidSizeIndex = 0;asteroidPos = [3,0,2];asteroidVel = [0, 0, 0];}
// update visual scale
asteroidVisualScale = asteroidSizeLevels[asteroidSizeIndex];}
const asteroid = createSurface(
0, 2 * Math.PI, 40, 0, Math.PI, 20,
(phi, theta) => Math.sin(theta) * Math.cos(phi),
(phi, theta) => Math.sin(theta) * Math.sin(phi),
(phi, theta) => Math.cos(theta)
);for (let i = 0; i < asteroid[0].length; i++) {
asteroid[0][i] *= asteroidRadius;}
// Add asteroid object to scene
const asteroidObj = addObject(asteroid, [0.35, 0.2, 0.05], 1.0);
// -----------------------------------------------------------------------------
// Tiny fragments for asteroid impact
// -----------------------------------------------------------------------------
const fragmentGeom = JSON.parse(JSON.stringify(asteroid)); // same shape
for (let i = 0; i < fragmentGeom[0].length; i++) {
fragmentGeom[0][i] *= 0.2;} // shrink to 20% size
const frag1Obj = addObject(fragmentGeom, [0.35, 0.2, 0.05], 0.0);
const frag2Obj = addObject(fragmentGeom, [0.35, 0.2, 0.05], 0.0);
let frag1Pos = [0,0,0], frag2Pos = [0,0,0];
let frag1Vel = [0,0,0], frag2Vel = [0,0,0];
let fragTimer = 0;
function updateFragments(dt) {
if (fragTimer <= 0) return;
fragTimer -= dt;
const fade = Math.max(0, fragTimer / 1.0); // fade over 1s
frag1Pos = add(frag1Pos, mul(frag1Vel, dt));
frag2Pos = add(frag2Pos, mul(frag2Vel, dt));
frag1Obj.opacity = fade;
frag2Obj.opacity = fade;
}
// -----------------------------------------------------------------------------
// Helper: Update asteroid by translating all vertices
// -----------------------------------------------------------------------------
function updateBasketballAsteroid(dt) {
// ----------------------------
// Gravity toward origin
// ----------------------------
if (gravityOn) {
const dir = normalize(mul(asteroidPos, -1));
const dist = Math.hypot(...asteroidPos);
const g = gravityStrength * dist * dt * 60 * 0.1;
asteroidVel = add(asteroidVel, mul(dir, g));
}
// ----------------------------
// Integrate motion
// ----------------------------
asteroidPos = add(asteroidPos, mul(asteroidVel, dt));
// ----------------------------
// Green sphere collision (exactly like ship)
// ----------------------------
if (greenSphereOn) {
const d = Math.hypot(...asteroidPos);
if (d > 3.5) {asteroidrefreshready=true}
if (d < greenRadius) {
const dir = normalize(asteroidPos);
asteroidPos = mul(dir, greenRadius);
const inward = asteroidVel[0]*dir[0]
          + asteroidVel[1]*dir[1]
          + asteroidVel[2]*dir[2];
if (inward < 0) {if (asteroidrefreshready) {playThud(); asteroidrefreshready=false}
asteroidVel = sub(asteroidVel, mul(dir, inward));}}}
// ----------------------------
// Blue sphere interaction
// ----------------------------
const dBlue = Math.hypot(...asteroidPos);
if (sphereObj.opacity >= 1.0 && dBlue < blueBaseRadius) {
const penetration = blueBaseRadius - dBlue;
const dir = normalize(asteroidPos);
asteroidVel = add(asteroidVel, mul(dir, penetration * 20 * dt));
asteroidPos = add(asteroidPos, mul(dir, penetration * 0.5));
if (asteroidblueready) {playBounce(); asteroidblueready=false}}
if (asteroidHoopCooldown <= 0 && checkHoopCollision(asteroidPos)) {
asteroidVel = mul(asteroidVel, -1); playBounce()
asteroidHoopCooldown = HOOP_COOLDOWN_TIME;}
// Bounding box constraint for asteroid (elastic bounce)
for (let i = 0; i < 3; i++) {
if (asteroidPos[i] > boundLimit - asteroidRadius) {
asteroidPos[i] = boundLimit - asteroidRadius;
if (asteroidVel[i] > 0) {asteroidVel[i] *= -0.95; playfuturebounce(); playfuturebounce2; playBounce} // bounce back with damping
} else if (asteroidPos[i] < -boundLimit + asteroidRadius) {
asteroidPos[i] = -boundLimit + asteroidRadius;
if (asteroidVel[i] < 0) asteroidVel[i] *= -0.95;}}
asteroidVel = mul(asteroidVel, asteroidDrag);} //drag
// -----------------------------------------------------------------------------
// NEW: Toggle and state variables for interactive spheres
// -----------------------------------------------------------------------------
let greenSphereOn = false;      // toggled by 'p'
let transparentBlue = true;     // blue sphere currently translucent? (toggled by 'o')
const greenRadius = 2.9;        // match inner sphere radius
const blueBaseRadius = 3.0;     // for force computations
const keys = {};
// ONE unified keydown listener
document.addEventListener("keydown", e => {
const k = e.key.toLowerCase();
keys[k] = true;
// toggle wireframe
if (k === 'e') {
showWireframe = !showWireframe;
playtick()}
// toggle sphere opacity
if (k === 'o') {
sphereObj.opacity = (sphereObj.opacity < 1.0) ? 1.0 : 0.65;
playtick2()}
// toggle camera mount
if (k === 'm') {cameraMounted = !cameraMounted;console.log("Camera mounted:", cameraMounted);
playfuturebounce2(); playtick2(); playtick()}
// wave triggers(
if (k === 't') {playBounce(); playtick(); playfuturebounce()
const [, , , , , netField, nxnet, nynet] = netGeom;
netField[Math.floor(nxnet / 2)][Math.floor(nynet/ 2)][0] = 5;
const [, , , , , mobField, nxmob, nymob] = mobius;
mobField[Math.floor(nxmob / 2)][Math.floor(nymob/ 2)][0] = 1.1;
// existing surface tap
applyImpulseToSphere(10, 2);
// --- NEW: tangential kick to asteroid ---
const radial = normalize(asteroidPos);     // gravity direction
const tangential = perpendicularTo(radial);
const impulseMag = Math.hypot(...laserVel) * 1.2;
applyImpulseToAsteroid(tangential, impulseMag);}
if (k === 'f' && !laserActive) {playChirp()
// snapshot ship state to avoid mutating shared references
const shipRot = quatToMatrix([...shipQuat]);
const forward = normalize([-shipRot[8], -shipRot[9], -shipRot[10]]);
const shipForwardPos = add([...shipPos], mul(forward, 0.6));
laserQuat = [...shipQuat];  // deep copy
laserPos = shipForwardPos.slice();
laserVel = mul(forward, 15.0);
laserLife = 1.0;
laserActive = true;
laserObj.opacity = 1.0;}
if (k === 'r') resetSphere();
if (k === 'v') basketball2hoop();
// toggle green sphere
if (k === 'p') {
playfuturebounce(); playtick()
greenSphereOn = !greenSphereOn; innerSphereObj.opacity = greenSphereOn ? 1.0 : 0.0;
console.log("Green sphere:", greenSphereOn ? "ON" : "OFF");}
// toggle gravity field
if (k === 'g') {gravityOn = !gravityOn;
console.log("Gravity:", gravityOn ? "ON" : "OFF"); playFieldEffect()}
// orient ship and camera toward origin
if (k === 'l') {orientTowardOrigin(); playtick(); playtick2()}
if (k === 'q') {resetSimulation();}});
// ONE unified keyup listener
document.addEventListener("keyup", e => {const k = e.key.toLowerCase(); keys[k] = false;});
// -----------------------------------------------------------------------------
// Sphere Perturbation Helpers
// -----------------------------------------------------------------------------
function applyImpulseToSphere(amplitude = 3, radius = 5) {
const [verts, norms, inds, c2i, i2c, field, nx, ny] = sphere;
const cx = Math.floor(nx / 2);
const cy = Math.floor(ny / 2);
for (let i = 0; i < nx; i++) {for (let j = 0; j < ny; j++) {
// Compute distance on sphere parameter grid
const dx = i - cx;
const dy = j - cy;
const dist = Math.sqrt(dx * dx + dy * dy);
if (dist < radius) {
const falloff = Math.cos((dist / radius) * Math.PI) * 0.5 + 0.5;
field[i][j][1] += amplitude * falloff; }}}} // add velocity impulse
function resetSphere() {
playReset()
const [, , , , , field, nx, ny] = sphere;
for (let i = 0; i < nx; i++) {
for (let j = 0; j < ny; j++) {
field[i][j][0] = 0;
field[i][j][1] = 0; }}
const [, , , , , fieldnet, nxnet, nynet] = netGeom;
for (let i = 0; i < nxnet; i++) {
for (let j = 0; j < nynet; j++) {
fieldnet[i][j][0] = 0;
fieldnet[i][j][1] = 0;}}
const [, , , , , fieldmob, nxmob, nymob] = mobius;
for (let i = 0; i < nxmob; i++) {for (let j = 0; j < nymob; j++) {
fieldmob[i][j][0] = 0; fieldmob[i][j][1] = 0;}}
asteroidSizeIndex=0
asteroidVisualScale = asteroidSizeLevels[asteroidSizeIndex];}
function basketball2hoop(){
playFieldEffect()
asteroidPos=[-0.45,0.25,18.2]
asteroidVel=[0,0,0]
asteroidSizeIndex=0
asteroidVisualScale = asteroidSizeLevels[asteroidSizeIndex];}
function convolaplace(image, boundary = 1) {
const laplaceKernel = [
[0.25, 0.5, 0.25],
[0.5, -3.0, 0.5],
[0.25, 0.5, 0.25]];
const nx = image.length;
const ny = image[0].length;
const output = new Array(nx).fill().map(() => new Array(ny).fill(0));
function getPixel(x, y) {
if (x < 0 || y < 0 || x >= nx || y >= ny) {
if (boundary === 0) return 0; // zero padding
if (boundary === 1) { // extend (replicate)
x = Math.max(0, Math.min(nx - 1, x));
y = Math.max(0, Math.min(ny - 1, y));}
if (boundary === -1) { // wrap around
x = (x + nx) % nx;
y = (y + ny) % ny;}}
return image[x][y];}
for (let i = 0; i < nx; i++) {for (let j = 0; j < ny; j++) {
let sum = 0;
for (let ki = -1; ki <= 1; ki++) {for (let kj = -1; kj <= 1; kj++) {
const val = getPixel(i + ki, j + kj);
sum += laplaceKernel[ki + 1][kj + 1] * val;}}
output[i][j] = sum;}}
return output;}
const innerSphere = createSurface(0,2*Math.PI,20,0,Math.PI,10,
(phi,theta)=>2.9*Math.sin(theta)*Math.cos(phi),
(phi,theta)=>2.9*Math.sin(theta)*Math.sin(phi),
(phi,theta)=>2.9*Math.cos(theta));
const innerSphereObj = addObject(innerSphere, [0.8,1.0,0.7], 0.0); // start invisible
function resetSimulation() {
shipPos = [2.5, 2.5, 2.5];
shipVel = [0, 0, 0];
asteroidPos=[3,0,2]; //[3,0,2]
asteroidVel=[0,0,0];
shipQuat = vec2quat([0, 0, 1]);
camPos = [5, 5, 5];
camQuat = vec2quat([-1, -1, -1]);
cameraMounted=false;
resetSphere();
console.log("Simulation reset!");
gravityOn=false
greenSphereOn=true
innerSphereObj.opacity = greenSphereOn ? 1.0 : 0.0;}
function dot(a, b) {
return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]; }
function perpendicularTo(v) {
// pick an arbitrary axis not parallel to v
const a = Math.abs(v[0]) < 0.9 ? [1,0,0] : [0,1,0];
return normalize(cross(v, a));}
// ------------------------------------------------------------
// Helper: instantaneous impulse applied to asteroid
// ------------------------------------------------------------
const asteroidMass = 5.0; // bigger = heavier, tune by feel
function applyImpulseToAsteroid(impulseVec) {
// Δv = J / m
const dv = mul(impulseVec, 1 / asteroidMass);
asteroidVel = add(asteroidVel, dv);}
const shipRadius = 0.05; // tweak until it "feels" right
function applyImpulseToSphereAtPoint(worldPos, amplitude = 5, radius = 0.6) {
const [verts, , , , , field, nx, ny] = sphere;
for (let i = 0; i < nx; i++) {
for (let j = 0; j < ny; j++) {
const idx = i * ny + j;
const p = [verts[3*idx], verts[3*idx+1], verts[3*idx+2]];
const d = Math.hypot(p[0] - worldPos[0], p[1] - worldPos[1], p[2] - worldPos[2]);
if (d < radius) {
const falloff = Math.cos((d / radius) * Math.PI) * 0.5 + 0.5; field[i][j][1] += amplitude * falloff;}}}}
// -----------------------------------------------------------------------------
// SUN VISUAL (for the lighting direction)
// -----------------------------------------------------------------------------
const sunDir = normalize([-1, -1, 1]); // same as shader L
const sunDistance = 31;
const sunPos = mul(sunDir, -sunDistance);
// Small bright sphere (sun core)
const sunCoreGeom = createSurface(0, 2*Math.PI, 40, 0, Math.PI, 20,
(phi,theta)=>2*Math.sin(theta)*Math.cos(phi),
(phi,theta)=>2*Math.sin(theta)*Math.sin(phi),
(phi,theta)=>2*Math.cos(theta));
const sunCoreObj = addObject(sunCoreGeom, [1.0, 0.9, 0.3], 1.0);
sunCoreObj.position = sunPos;


// Slightly larger transparent halo
const sunHaloGeom = createSurface(0, 2*Math.PI, 40, 0, Math.PI, 20,
(phi,theta)=>2.5*Math.sin(theta)*Math.cos(phi),
(phi,theta)=>2.5*Math.sin(theta)*Math.sin(phi),
(phi,theta)=>2.5*Math.cos(theta));
const sunHaloObj = addObject(sunHaloGeom, [1.0, 0.9, 0.3], 0.2);
sunHaloObj.position = sunPos;
// -----------------------------------------------------------------------------
// BASKETBALL HOOP ASSEMBLY
// -----------------------------------------------------------------------------
const hoopRadius = 1.4;
const hoopThickness = 0.25;
const backboardWidth = 4.0;
const backboardHeight = 3.0;
// --- Torus (orange rim) ---
function createTorus(R, r, nMajor, nMinor) {
const verts = [], norms = [], inds = [];
for (let i = 0; i < nMajor; i++) {
const phi = (i / nMajor) * 2 * Math.PI;
for (let j = 0; j < nMinor; j++) {
const theta = (j / nMinor) * 2 * Math.PI;
const x = (R + r * Math.cos(theta)) * Math.cos(phi);
const z = (R + r * Math.cos(theta)) * Math.sin(phi);
const y = r * Math.sin(theta);
verts.push(x, y, z);
norms.push(Math.cos(theta) * Math.cos(phi), Math.cos(theta) * Math.sin(phi), Math.sin(theta));}}
for (let i = 0; i < nMajor; i++) {for (let j = 0; j < nMinor; j++) {
const a = i * nMinor + j;
const b = ((i + 1) % nMajor) * nMinor + j;
const c = ((i + 1) % nMajor) * nMinor + ((j + 1) % nMinor);
const d = i * nMinor + ((j + 1) % nMinor);
inds.push(a, b, d, b, c, d);}}
return [verts, norms, inds];}
const torusGeom = createTorus(hoopRadius, hoopThickness, 40, 15);
const hoopObj = addObject(torusGeom, [1.0, 0.4, 0.0], 1.0);
hoopObj.position = [0, 0-1, boundLimit - 0.1-1.65]; // inside the +Z face
// --- Backboard (white rectangle) ---
const backboardGeom = createSurface(-backboardWidth/2, backboardWidth/2, 2,
-backboardHeight/2, backboardHeight/2, 2, (x,y)=>x, (x,y)=>y, (x,y)=>boundLimit - 0.05);
const backboardObj = addObject(backboardGeom, [1.0, 1.0, 1.0], 1.0);
// --- Translucent net cylinder ---
const netGeom = createSurface(0, 2*Math.PI, 40, 0, 1.5, 20,
(phi,h)=>hoopRadius*Math.sin(phi),
(phi,h)=>boundLimit - 0.1 - h - 20.85,
(phi,h)=>hoopRadius*Math.cos(phi)+18.9-0.8);
const netObj = addObject(netGeom, [0.4, 0.4, 0.4], 0.5);
// --- mobius strip
const mobius = createSurface(0, 2*Math.PI, 40, -0.07, 0.07, 10,
(u,v)=>(1+(v/2)*Math.cos(u))*Math.cos(u)*10,
(u,v)=>(1+(v/2)*Math.cos(u))*Math.sin(u)*10,
(u,v)=>(v/2)*Math.sin(u/2)*0.3);
const mobiusObj = addObject(mobius, [1, 0.3, 0.5], 0.3);
// --- redboard (red rectangle) ---
const redboard = createSurface(-1.0, 1.0, 2,-1.05, 0.4, 2, (x,y)=>x, (x,y)=>y, (x,y)=>boundLimit - 0.1);
const redboardobj= addObject(redboard, [0.9, 0.3, 0.2], 1);
// --- redboard (red rectangle) ---
const whiteboard = createSurface(-0.7, 0.7, 2,-0.8, 0.1, 2, (x,y)=>x, (x,y)=>y, (x,y)=>boundLimit - 0.15);
const whiteboardobj= addObject(whiteboard, [1, 1, 1], 1);
// -----------------------------------------------------------------------------
// Hoop collision cooldowns (seconds)
// -----------------------------------------------------------------------------
let shipHoopCooldown = 0;
let asteroidHoopCooldown = 0;
const HOOP_COOLDOWN_TIME = 1.0; // seconds
let shipboundbouncecooldown = 0;
const backboardcooldowntime=0.2
let shipbackboardcooldown = 0;
let shiprefreshready = true;
let asteroidrefreshready = true;
let shipblueready = true;
let asteroidblueready = true;
function checkHoopCollision(pos) {
const hoopCenter = [0, -1, 18.2];
const rel = sub(pos, hoopCenter);
// Distance in hoop plane (ZX)
const radialDist = Math.hypot(rel[0], rel[2]);
// Distance out of hoop plane (Z)
const dz = rel[1];
// 1) Must be near the hoop plane
if (Math.abs(dz) > 0.12) return false;
// 2) Must be near the ring radius (NOT the center hole)
if (Math.abs(radialDist - hoopRadius) > 1 ) return false;
return true;}
function checkbackboardcollision(pos) {
const backboardcenter = [0, 0, boundLimit - 0.03];
const rel = sub(pos, backboardcenter);
// Distance in hoop plane (XY)
const radialDist = Math.hypot(rel[0], rel[1]);
// Distance out of hoop plane (Z)
const dz = rel[2];
if (Math.abs(dz) > 0.12) return false;
if (Math.abs(radialDist) > 5.0 ) return false;
return true;}
// -----------------------------------------------------------------------------
// MAIN LOOP --MAIN SIMULATION HEARTBEAT LIVE LOOP HERE
// -----------------------------------------------------------------------------
let lastTime = performance.now();
let initialized = false;
const c = 2;
const damping = 0.99999;
const capAmp = 20;
const capVel = 20;
const amplitudeScale = 5;
function clamp(v, min, max) { return Math.max(min, Math.min(max, v)); }


function loop(time) {
const dt = Math.min((time - lastTime) / 1000, 0.03);
lastTime = time;
shipHoopCooldown -= dt;
asteroidHoopCooldown -= dt;
shipbackboardcooldown -= dt;
shipboundbouncecooldown -= dt;
if (!initialized) {playReset()
// Initialize sphere
const [, , , , , sphereField, nx2, ny2] = sphere;
for (let i = 0; i < nx2; i++)
for (let j = 0; j < ny2; j++)
sphereField[i][j] = [0, 0];
sphereField[Math.floor(nx2 / 2)][Math.floor(ny2 / 2)][0] = 5;
// Initialize net
const [, , , , , netField, nxnet, nynet] = netGeom;
for (let i = 0; i < nxnet; i++)
for (let j = 0; j < nynet; j++)
netField[i][j] = [0, 0];
netField[Math.floor(nxnet / 2)][Math.floor(nynet/ 2)][0] = 25;
// Initialize mob
const [, , , , , mobField, nxmob, nymob] = mobius;
for (let i = 0; i < nxmob; i++)
for (let j = 0; j < nymob; j++)
mobField[i][j] = [0, 0];
mobField[Math.floor(nxmob / 2)][Math.floor(nymob/ 2)][0] = 25;
initialized = true;}
// --------------------------------------------------------
// Sphere wave evolution (periodic boundaries)
// --------------------------------------------------------
const [v2, , , , , sphereField, nx2, ny2] = sphere;
const u2 = sphereField.map(row => row.map(v => v[0]));
const lap2 = convolaplace(u2, -1); // periodic boundaries!
let lambda=0.2
for (let i = 0; i < nx2; i++) {for (let j = 0; j < ny2; j++) {
const du2 = 10* c* c * lap2[i][j]-lambda * sphereField[i][j][0];
sphereField[i][j][1] = clamp((sphereField[i][j][1] + du2 * dt), -capVel, capVel);
sphereField[i][j][0] = clamp(sphereField[i][j][0] + sphereField[i][j][1] * dt, -capAmp*0 -3, capAmp);}}
// Apply displacement radially to sphere
for (let i = 0; i < nx2; i++) {for (let j = 0; j < ny2; j++) {
const idx = i * ny2 + j;
const phi = (i / (nx2 - 1)) * 2 * Math.PI;
const theta = (j / (ny2 - 1)) * Math.PI;
const r = 3 + 0.5 * sphereField[i][j][0]; // radial wave deformation
v2[3 * idx + 0] = r * Math.sin(theta) * Math.cos(phi);
v2[3 * idx + 1] = r * Math.sin(theta) * Math.sin(phi);
v2[3 * idx + 2] = r * Math.cos(theta);}}
recomputeNormals(sphere, nx2, ny2);
// -----------------------------------------------------------------------------
// Blue sphere physical response — transparent vs opaque
// -----------------------------------------------------------------------------
const shipDist = Math.hypot(...shipPos);
const shipDir  = normalize(shipPos);
const cutoff    = 0.8;      // local influence radius (in world units)
const attraction = 100.0;    // how strongly fluid surface pulls toward ship
const repulsion  = 20.0;    // how strong the hard barrier pushes back
const baseR      = blueBaseRadius;
if (sphereObj.opacity < 1.0) {
// -----------------------------
// Transparent: "liquid" behavior
// -----------------------------
for (let i = 0; i < nx2; i++) {for (let j = 0; j < ny2; j++) {
const idx = i * ny2 + j;
const vx = v2[3 * idx + 0];
const vy = v2[3 * idx + 1];
const vz = v2[3 * idx + 2];
const p = [vx, vy, vz];
const d = Math.hypot(p[0]-shipPos[0], p[1]-shipPos[1], p[2]-shipPos[2]);
if (d < cutoff) {
// Compute direction from surface point to ship (radial constraint)
const radialDir = normalize(p);
const towardShip = sub(shipPos, p);
const align = Math.max(0, (radialDir[0]*towardShip[0] + radialDir[1]*towardShip[1] + radialDir[2]*towardShip[2]) / Math.hypot(...towardShip));
// Local "suction" perturbation in field velocity
const strength = (1 - d / cutoff) * align * attraction;
sphereField[i][j][1] -= strength * dt;}}}} else {
// -----------------------------
// Opaque: "solid" behavior
// -----------------------------
if (shipDist < baseR) {
const penetration = baseR - shipDist;
const repelForce = repulsion * penetration;
shipVel = add(shipVel, mul(shipDir, repelForce * dt));
shipPos = add(shipPos, mul(shipDir, penetration * 0.5));
if (shipblueready) {playBounce(); shipblueready=false}
// push nearest surface region slightly outward visually
let minI = 0, minJ = 0, minDist = Infinity;
for (let i = 0; i < nx2; i++) {
for (let j = 0; j < ny2; j++) {
const idx = i * ny2 + j;
const vx = v2[3 * idx + 0];
const vy = v2[3 * idx + 1];
const vz = v2[3 * idx + 2];
const d = Math.hypot(vx-shipPos[0], vy-shipPos[1], vz-shipPos[2]);
if (d < minDist) { minDist = d; minI = i; minJ = j; }}}
if (minDist < cutoff) { // small bump outward
sphereField[minI][minJ][0] -= 2.0 * (1 - minDist / cutoff);}
} else {shipblueready=true}}
// =====================================================================
// BASKETBALL =>  BLUE SPHERE SURFACE (LIQUID PLANET RESPONSE)
// =====================================================================
let basketballcutoff=0.4
if (sphereObj.opacity < 1.0) {
for (let i = 0; i < nx2; i++) {for (let j = 0; j < ny2; j++) {
const idx = i * ny2 + j;
const vx = v2[3 * idx + 0];
const vy = v2[3 * idx + 1];
const vz = v2[3 * idx + 2];
const p = [vx, vy, vz];
const d = Math.hypot(p[0] - asteroidPos[0],p[1] - asteroidPos[1],p[2] - asteroidPos[2]);
if (d < basketballcutoff) {
const radialDir = normalize(p);
const towardBall = sub(asteroidPos, p);
const align = Math.max(0,dot(radialDir, towardBall) / Math.hypot(...towardBall));
// Weaker than ship so it looks sensible
const strength =(1 - d / basketballcutoff) * align * attraction * 0.4;
sphereField[i][j][1] -= strength * dt;}}}}
// =====================================================================
// BASKETBALL upon BLUE SPHERE SURFACE (SOLID PLANET RESPONSE)
// =====================================================================
if (Math.hypot(...asteroidPos) > 5) {asteroidblueready=true}
if (sphereObj.opacity >= 1.0) {
const asteroidDist = Math.hypot(...asteroidPos);
const asteroidDir  = normalize(asteroidPos);
if (asteroidDist < baseR + asteroidRadius) {
const penetration = baseR + asteroidRadius - asteroidDist;
const repelForce  = repulsion * penetration * 0.5;
// Push basketball outward
asteroidVel = add(asteroidVel,mul(asteroidDir, repelForce * dt * 0.01111));
asteroidPos = add(asteroidPos,mul(asteroidDir, penetration * 0.5 * 0.01111));
// Visual dent on planet surface
let minI = 0, minJ = 0, minDist = Infinity;
for (let i = 0; i < nx2; i++) {for (let j = 0; j < ny2; j++) {
const idx = i * ny2 + j;
const vx = v2[3 * idx + 0];
const vy = v2[3 * idx + 1];
const vz = v2[3 * idx + 2];
const d = Math.hypot(
 vx - asteroidPos[0],
 vy - asteroidPos[1],
 vz - asteroidPos[2]
); if (d < minDist) {minDist = d; minI = i; minJ = j;}}}
if (minDist < cutoff) {sphereField[minI][minJ][0] -=2.5 * (1 - minDist / cutoff);
if (asteroidblueready) {playBounce(); asteroidblueready=false}}}}
// --- Flame flicker ---
const t = performance.now() * 0.005;   // time in seconds
const base = 1;                      // base opacity
const amp  = 0.3;                      // flicker amplitude
flameObj.opacity = base + amp * Math.abs(Math.sin(t * 3.0 + Math.sin(t * 1.3)));
flameObj.color = [1.0,0.5 + 0.5 * (0.5 + 0.5 * Math.sin(t * 3.0)),
0.0 + 0.5 * (0.5 + 0.5 * Math.sin(t * 3.0)) * 0.2];
recomputeFlatNormals(shipObj.geom[0], shipObj.geom[2], shipObj.geom[1]);
recomputeFlatNormals(flameObj.geom[0], flameObj.geom[2], flameObj.geom[1]);
//asteroid basketball dynamics here vv
updateBasketballAsteroid(dt); updateFragments(dt);
// ------------------------------------------------------------
// Laser update
// ------------------------------------------------------------
if (laserActive) {
// --- gravity affects laser ---
if (gravityOn) {
const dir = normalize(mul(laserPos, -1));
const dist = Math.hypot(...laserPos);
const g = gravityStrength * dist * dt * 60 * 0.3;
laserVel = add(laserVel, mul(dir, g));
}
laserPos = add(laserPos, mul(laserVel, dt));
laserLife -= dt;
// collision with basketball
const d = Math.hypot(
laserPos[0] - asteroidPos[0],laserPos[1] - asteroidPos[1],laserPos[2] - asteroidPos[2]);
if (d < asteroidRadius) {playThud() // impulse transfer
applyImpulseToAsteroid(mul(normalize(laserVel), 3.0));
// -----------------------------------------------------------------------------
// Asteroid hit effect: spawn two flying fragments
// -----------------------------------------------------------------------------
fragTimer = 1.0; // 1 second lifetime
frag1Pos = asteroidPos.slice();
frag2Pos = asteroidPos.slice();
const randDir = normalize([Math.random()*2 - 1,Math.random()*2 - 1,Math.random()*2 - 1]);
frag1Vel = mul(randDir, 4.0);
frag2Vel = mul(randDir, -4.0);
frag1Obj.opacity = 1.0;
frag2Obj.opacity = 1.0;
handleAsteroidHit();
console.log("Basketball Asteroid Basketroid hit! Size =", asteroidVisualScale);
laserActive = false;
laserObj.opacity = 0.0;}
if (laserLife <= 0) {
laserActive = false;
laserObj.opacity = 0.0;}
// ------------------------------------------------------------
// collision with blue sphere surface
// ------------------------------------------------------------
const dBlue = Math.hypot(...laserPos);
if (dBlue < blueBaseRadius + 0.05) {
const hitDir = normalize(laserPos);
if(sphereObj.opacity < 1) {playBounce()} else {playfuturebounce()}
if (sphereObj.opacity < 1.0) { // ---------- LIQUID ----------
applyImpulseToSphereAtPoint(laserPos, 10.0, 0.6);} else { // --- SOLID ----
applyImpulseToSphereAtPoint(laserPos, 8.0, 0.4);}
laserActive = false;
laserObj.opacity = 0.0;}}
// --------------------------------------------------------
// Update scene
// --------------------------------------------------------
updateSpaceship(dt);
updateBuffers();
updateCamera();
drawScene();
requestAnimationFrame(loop);
// =====================================================================
// SHIP and ASTEROID COLLISION (impulse-based, duct-tape physics)
// =====================================================================
const delta = sub(shipPos, asteroidPos);
const dist  = Math.hypot(...delta);
const minDist = shipRadius + asteroidRadius;
if (dist < minDist && dist > 1e-6) {
// --- Collision normal (from asteroid to ship)
const n = normalize(delta);
// --- Relative velocity along normal
const relVel = sub(shipVel, asteroidVel);
const relNormalSpeed = dot(relVel, n);
// Only respond if they are moving toward each other
if (relNormalSpeed < 0) {
// --------------------------------------------------
// Impulse magnitude (inelastic billiards)
// --------------------------------------------------
const restitution = 0.6; // 0 = sticky, 1 = bouncy
const shipMass = 0.5;
const j =-(1 + restitution) * relNormalSpeed /(1 / shipMass + 1 / asteroidMass);
const impulse = mul(n, j);
// --- Apply impulse
playBounce()
applyImpulseToAsteroid(mul(impulse, -1)); // asteroid pushed away
shipVel = add(shipVel, mul(impulse, 1 / shipMass));
// --------------------------------------------------
// Positional correction (prevents sinking)
// --------------------------------------------------
const penetration = minDist - dist;
const correction = mul(n, penetration * 0.5);
shipPos     = add(shipPos, correction);
asteroidPos = sub(asteroidPos, correction);}}
//nothingbutnet ----------------
const [vnet, , , , , netField, nxnet, nynet] = netGeom;
const unet = netField.map(row => row.map(v => v[0]));
const lapnet = convolaplace(unet, -1); // periodic boundaries!
for (let i = 0; i < nxnet; i++) {
for (let j = 0; j < nynet; j++) {
const dunet = 10* c* c * lapnet[i][j]-lambda*3 * netField[i][j][0];
netField[i][j][1] = clamp((netField[i][j][1] + dunet * dt), -capVel, capVel);
netField[i][j][0] = clamp(netField[i][j][0] + netField[i][j][1] * dt, -2, 2);}}
// Apply displacement radially to cylinder
for (let i = 0; i < nxnet; i++) {for (let j = 0; j < nynet; j++) {
const idxnet = i * nynet + j;
const phinet = (i / (nxnet - 1)) * 2*Math.PI;
const thetanet = (j / (nynet - 1)) * 1.5;
const rnet = 1.35 + 0.2 * netField[i][j][0]; // radial wave deformation
vnet[3 * idxnet + 0] = rnet*Math.sin(phinet);
vnet[3 * idxnet + 1] = boundLimit-0.1-thetanet-20.85;
vnet[3 * idxnet + 2] = rnet * Math.cos(phinet)+18.9-0.8;}}
recomputeNormals(netGeom, nxnet, nynet);
const cutoffnet    = 0.8;      // local influence radius (in world units)
const attractionet = 100.0;    // how strongly net surface pulls toward ship
// -----------------------------
// Transparent: "liquid" behavior
// -----------------------------
for (let i = 0; i < nxnet; i++) {
for (let j = 0; j < nynet; j++) {
const idxnet = i * nynet + j;
const vxnet = vnet[3 * idxnet + 0];
const vynet = vnet[3 * idxnet + 1];
const vznet = vnet[3 * idxnet + 2];
const pnet = [vxnet, vynet, vznet];
const dnet = Math.hypot(pnet[0]-shipPos[0], pnet[1]-shipPos[1], pnet[2]-shipPos[2]);
if (dnet < cutoffnet) {
const strength = (1 - dnet / cutoffnet) * 1 * attractionet;
netField[i][j][1] -= strength * dt;}}}
// runbythemob ---------------------------------
const [vmob, , , , , mobField, nxmob, nymob] = mobius;
const umob = mobField.map(row => row.map(v => v[0]));
const lapmob = convolaplace(umob, -1); // periodic boundaries!
for (let i = 0; i < nxmob; i++) { for (let j = 0; j < nymob; j++) {
const dumob = 10* c* c * lapmob[i][j]-lambda*3 * mobField[i][j][0];
mobField[i][j][1] = clamp((mobField[i][j][1] + dumob * dt), -capVel, capVel);
mobField[i][j][0] = clamp(mobField[i][j][0] + mobField[i][j][1] * dt, -2, 2);}}
// Apply displacement radially to cylinder
for (let i = 0; i < nxmob; i++) {for (let j = 0; j < nymob; j++) {
const idxmob = i * nymob + j;
const phimob = (i / (nxmob - 1)) * 2*Math.PI;
const thetamob = ((j / (nynet - 1)) * 0.5);
//const rmob = 1.35 + 0.2 * mobField[i][j][0]; // radial wave deformation
vmob[3 * idxmob + 0] = (1+(thetamob/2)*Math.cos(phimob))*Math.cos(phimob)*10+ 1.5 * mobField[i][j][0];
vmob[3 * idxmob + 1] = (1+(thetamob/2)*Math.cos(phimob))*Math.sin(phimob)*10-(0.1*mobField[i][j][0]*mobField[i][j][0]*mobField[i][j][0]);
vmob[3 * idxmob + 2] = (thetamob/2)*Math.sin(phimob/2)*0.3 -0.2 * mobField[i][j][0];}}
recomputeNormals(mobius, nxmob, nymob);
const cutoffmob    = 2.3;      // local influence radius (in world units)
const attractionmob = 120.0;    // how strongly net surface pulls toward ship
for (let i = 0; i < nxmob; i++) {for (let j = 0; j < nymob; j++) {
const idxmob = i * nymob + j;
const vxmob = vmob[3 * idxmob + 0];
const vymob = vmob[3 * idxmob + 1];
const vzmob = vmob[3 * idxmob + 2];
const pmob = [vxmob, vymob, vzmob];
const dmob = Math.hypot(pmob[0]-shipPos[0], pmob[1]-shipPos[1], pmob[2]-shipPos[2]);
if (dmob < cutoffmob) {
const strength = (1 - dmob / cutoffmob) * 1 * attractionmob;
mobField[i][j][1] -= strength * dt;}}}
for (let i = 0; i < nxmob; i++) {
for (let j = 0; j < nymob; j++) {
const idxmob = i * nymob + j;
const vxmob = vmob[3 * idxmob + 0];
const vymob = vmob[3 * idxmob + 1];
const vzmob = vmob[3 * idxmob + 2];
const pmob = [vxmob, vymob, vzmob];
const dmob = Math.hypot(pmob[0]-asteroidPos[0], pmob[1]-asteroidPos[1], pmob[2]-asteroidPos[2]);
if (dmob < cutoffmob) {const strength = (1 - dmob / cutoffmob) * 1 * attractionmob;
mobField[i][j][1] -= strength * dt;}}}}; loop(0);



}




