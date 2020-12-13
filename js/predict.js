let originalVideoWidth=640;
let model;
const CLASSES =  ({0:'Ideal',1:'Not'})
const imageScaleFactor = 0.2;
const outputStride = 16;
const flipHorizontal = false;
const contentWidth = 800;
const contentHeight = 600;

var rect = {}
loadModel();
async function loadModel(){
  console.log("model loading..");

  $("#console").html(`<li>model loading...</li>`);
  model = await tf.loadModel(`../vgg16_model/model.json`);

  console.log("model loaded.");
  $("#console").html(`<li>model loaded.</li>`);
};

bindPage();

async function bindPage() {
    const net = await posenet.load();
    let video;
    try {
        video = await loadVideo();
    } catch(e) {
        console.error(e);
        return;
    }
    detectPoseInRealTime(video, net);
}

async function loadVideo() {
    const video = await setupCamera();
    video.play();
    return video;
}

async function setupCamera() {
    const video = document.getElementById('video');
    if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
        const stream = await navigator.mediaDevices.getUserMedia({
            'audio': false,
            'video': true});
        video.srcObject = stream;

        return new Promise(resolve => {
            video.onloadedmetadata = () => {
                resolve(video);
            };
        });
    } else {
        const errorMessage = "This browser does not support video capture, or this device does not have a camera";
        alert(errorMessage);
        return Promise.reject(errorMessage);
    }
}

function detectPoseInRealTime(video, net) {
    const canvas = document.getElementById('canvas');
    const ctx = canvas.getContext('2d');
    const flipHorizontal = true; // since images are being fed from a webcam

    async function poseDetectionFrame() {
        let poses = [];
        const pose = await net.estimateSinglePose(video, imageScaleFactor, flipHorizontal, outputStride);
        poses.push(pose);

        ctx.clearRect(0, 0, contentWidth,contentHeight);

        ctx.save();
        ctx.scale(-1, 1);
        ctx.translate(-contentWidth, 0);
        ctx.drawImage(video, 0, 0, contentWidth, contentHeight);
        ctx.restore();
   poses.forEach(({ s, keypoints }) => {
   drastroke(keypoints[5],keypoints[6],ctx),drastroke(keypoints[6],keypoints[12],ctx),drastroke(keypoints[5],keypoints[11],ctx), drastroke(keypoints[11],keypoints[12],ctx),
   get_rect(keypoints[5],keypoints[12], ctx);
 });
        predict(rect);
        console.log(rect);
        requestAnimationFrame(poseDetectionFrame);
    }
    poseDetectionFrame();
}

/*function drawWristPoint(wrist,ctx){
    ctx.beginPath();
    ctx.arc(wrist.position.x , wrist.position.y, 3, 0, 2 * Math.PI);
    ctx.fillStyle = "pink";
    ctx.fill();
}*/

function drastroke(begin,finish,ctx){
    ctx.beginPath();
    ctx.moveTo(begin.position.x, begin.position.y);
    ctx.lineTo(finish.position.x, finish.position.y);
    ctx.strokeStyle = 'red';
    ctx.stroke();
}

function get_rect(begin, finish, ctx){
  let bx = begin.position.x;
  let by = begin.position.y;
  let fx = finish.position.x;
  let fy = finish.position.y;
  rect.x = bx;
  rect.y = by;
  rect.width = bx-fx;
  rect.height = by-fy;
}

async function predict(rect){

  let tensor = captureWebcam(rect) ;

  let prediction = await model.predict(tensor).data();
  let results = Array.from(prediction)
              .map(function(p,i){
  return {
      probability: p,
      className: CLASSES[i],
      };
  }).sort(function(a,b){
      return b.probability-a.probability;
  }).slice(0,2);

  $("#console").empty();
  results.forEach(function(p){
    $("#console").append(`<li>${p.className} : ${p.probability.toFixed(5)}</li>`);
		console.log(p.className,p.probability.toFixed(5))

});
};

function captureWebcam(rect) {
  var faceCanvas = $('#faceCanvas').get(0);
  var faceContext = faceCanvas.getContext('2d');

  //adjust original video size
  var adjust = originalVideoWidth / video.width
  faceContext.drawImage(video, rect.x * adjust , rect.y * adjust, rect.width * adjust, rect.height * adjust,0, 0, 100, 100);

	tensor_image = preprocessImage(faceCanvas);

	return tensor_image;
}

function preprocessImage(image){
  const channels = 3;
  let tensor = tf.browser.fromPixels(image, channels).resizeNearestNeighbor([100,100]).toFloat();
  let offset = tf.scalar(255);
  return tensor.div(offset).expandDims();
};
