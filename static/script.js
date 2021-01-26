const serverOrigin = window.location.origin
const host = window.location.host
const socket = io();

socket.on('connect', () => {
    socket.emit('events', {data: 'Connected to the SocketServer...'});
});

socket.on('my_response', function (msg, cb) {
    console.log(msg)
});

function fileSelected(filesInput) {
    const video = document.createElement('video');
    video.preload = 'metadata';

    video.onloadedmetadata = () => {
        window.URL.revokeObjectURL(video.src);
        if (video.duration > 25) {
            addNotification('Selected video has too long duration... Should be no more than 25 seconds');
            document.getElementById('search-btn').disabled = false
            filesInput.value = ''
            return;
        }

        document.getElementById('search-btn').disabled = false
    }

    video.src = URL.createObjectURL(filesInput.files[0]);
}

function searchVideo() {
    const files = document.getElementById('video-file').files;

    if (files.length === 0) {
        console.warn('No video file selected...')
        return
    }

    const formData = new FormData();
    formData.append("input_file", files[0]);

    $('.alert')[0].hidden = true
    $('#search-btn')[0].style.display = 'none'
    $('#search-btn-loading')[0].style.display = 'inline-block'

    fetch(serverOrigin + '/api/recognize', {method: 'POST', body: formData})
        .then(resp => resp.json())
        .then(respJsonData => {

            $('#search-btn')[0].style.display = 'inline-block'
            $('#search-btn-loading')[0].style.display = 'none'
            const alert = $('.alert')[0]
            alert.hidden = false
            if (respJsonData.isFound) {
                alert.classList.add('alert-success')
                alert.classList.remove('alert-warning')
                alert.querySelector('h4').innerText = 'We found it!'
                alert.querySelector('p').innerText = `This fragment belongs to: ${respJsonData.name} . Elapsed time: ${Math.ceil(respJsonData.elapsedTime)} sec`
            } else {
                alert.classList.remove('alert-success')
                alert.classList.add('alert-warning')
                alert.querySelector('h4').innerText = 'Ooops...'
                alert.querySelector('p').innerText = `Sorry, we could not identify original video for given fragment...Elapsed time in seconds: ${Math.ceil(respJsonData.elapsedTime)}`
            }
        })
        .catch((error) => {
            console.log(error);
        })
}

function addNotification(message) {
    const newToast = `
        <div class="toast d-flex align-items-center" role="alert" aria-live="assertive" aria-atomic="true" data-bs-delay="10000">
            <div class="toast-body" style="text-align: left">${message}</div>
            <button type="button" class="btn-close ms-auto me-2" data-bs-dismiss="toast" aria-label="Close"></button>
        </div>
        `

    $('.toast-container')[0].innerHTML += newToast;
    $('.toast').toast('show');
}