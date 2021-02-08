fetch('/processed-videos', {method: 'GET'})
    .then(resp => resp.json())
    .then(videoInfoList => refreshFilesTable(videoInfoList))
    .catch(err => console.log(err));

function refreshFilesTable(videosList) {
    const tableBody = document.querySelector('#video-table tbody')
    while (tableBody.hasChildNodes()) {
        tableBody.removeChild(tableBody.lastChild);
    }

    videosList.forEach((videoInfo, index) => {
        const rowCnt = tableBody.rows.length;
        const tr = tableBody.insertRow(rowCnt);

        tr.insertCell(0).innerText = index;
        tr.insertCell(1).innerText = videoInfo.name;
        tr.insertCell(2).innerText = Math.floor(videoInfo.duration / 60) + 'min ' + (videoInfo.duration % 60) + 'sec'
        tr.insertCell(3).innerText = videoInfo.original_video_url
        const lastCell = tr.insertCell(4);
        const playBtn = document.createElement('button');
        playBtn.classList.add('btn');
        playBtn.classList.add('btn-outline-primary');
        playBtn.innerText = 'Play';
        playBtn.onclick = () => playVideo(videoInfo.original_video_url);
        lastCell.appendChild(playBtn);
    })
}

function playVideo(original_video_url) {
    const video = $('<video />', {
        id: 'original-video',
        src: '/video?path=' + original_video_url,
        type: 'video/mp4',
        controls: true,
        style: 'margin: 0 auto; width: 600px'
    });
    video.appendTo($('#video-play-container'));

    $('#video-modal')[0].style.display = 'block'
}

function closeModal() {
    $('#video-modal')[0].style.display = 'none'
    $('#original-video').remove();
}