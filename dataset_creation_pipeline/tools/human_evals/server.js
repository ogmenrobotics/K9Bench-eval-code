const express = require('express');
const fs = require('fs');
const app = express();
const PORT = 8090;

const videoFile = 'set_1.json';
const reviewedFile = 'human_eval_questions.json';

app.use(express.static('public'));
app.use(express.json());

app.get('/next-video', (req, res) => {
  const videos = JSON.parse(fs.readFileSync(videoFile, 'utf-8'));
  const reviewed = fs.existsSync(reviewedFile)
    ? JSON.parse(fs.readFileSync(reviewedFile, 'utf-8'))
    : [];

  const reviewedSet = new Set(
    reviewed.map(v => `${v.scene_name}|${v.question}`)
  );

  const next = videos.find(
    v => !reviewedSet.has(`${v.scene_name}|${v.question}`)
  );

  if (!next) return res.json({ video: null });

  res.json({
    video: {
      videoId: next.scene_name,
      question: next.question
    }
  });
});

app.post('/submit', (req, res) => {
  const { videoId, question, status, human_answer, reason } = req.body;

  const videos = JSON.parse(fs.readFileSync(videoFile, 'utf-8'));
  const meta = videos.find(v => v.scene_name === videoId && v.question === question);

  const entry = {
    idx: meta?.idx || null,
    dataset: meta?.dataset || null,
    scene_name: videoId,
    question_category: meta?.question_category || null,
    question,
    ground_truth: meta?.ground_truth || null,
    duration_sec: meta?.duration_sec || null,
    category: meta?.category || null,
    status,
    human_answer,
    reason: reason || '',
    human_evaluated: true,
    timestamp: new Date().toISOString()
  };

  let data = fs.existsSync(reviewedFile)
    ? JSON.parse(fs.readFileSync(reviewedFile, 'utf-8'))
    : [];

  const idx = data.findIndex(e => e.scene_name === videoId && e.question === question);
  if (idx !== -1) data[idx] = entry;
  else data.push(entry);

  fs.writeFileSync(reviewedFile, JSON.stringify(data, null, 2));
  res.json({ message: 'Saved' });
});

app.listen(PORT, () => {
  console.log(`Server running on http://localhost:${PORT}`);
});

