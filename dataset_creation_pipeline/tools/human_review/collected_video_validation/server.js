const express = require('express');
const fs = require('fs');
const path = require('path');
const app = express();
const PORT = 8000;

// Allow passing the file name from terminal: `node server.js my_video_ids.txt`
const videoFile = process.argv[2] || '../../data/meta_info/video_metadata/filtered_video_ids.txt';

const acceptedFile = '../../../data/meta_info/logs/yt_review_decisions/accepted_video_ids.txt';
const rejectedFile = '../../../data/meta_info/logs/yt_review_decisions/rejected_video_ids.txt';

app.use(express.static('public'));
app.use(express.json());

app.get('/next-video', (req, res) => {
    const accepted = fs.existsSync(acceptedFile)
        ? fs.readFileSync(acceptedFile, 'utf-8').split('\n')
        : [];

    const rejected = fs.existsSync(rejectedFile)
        ? fs.readFileSync(rejectedFile, 'utf-8').split('\n').map(line => line.split('|')[0])
        : [];

    const reviewed = new Set([...accepted, ...rejected]);

    const videos = fs.readFileSync(videoFile, 'utf-8')
        .split('\n')
        .filter(v => v.trim() && !reviewed.has(v.trim()));

    if (videos.length > 0) {
        res.json({ videoId: videos[0] });
    } else {
        res.json({ videoId: null });
    }
});

app.post('/submit', (req, res) => {
    const { videoId, status, reason } = req.body;

    if (status === 'accepted') {
        fs.appendFileSync(acceptedFile, `${videoId}\n`);
    } else if (status === 'rejected') {
        fs.appendFileSync(rejectedFile, `${videoId}|${reason}\n`);
    }

    res.json({ message: 'Saved' });
});

app.listen(PORT, () => {
    console.log(`Server running at http://localhost:${PORT}`);
    console.log(`Reading videos from: ${videoFile}`);
});
