import React, { useState } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import evaluationService from '../../services/evaluationService';

const EvaluationForm = () => {
    const location = useLocation();
    const navigate = useNavigate();
    const { task } = location.state || {};

    const [ratings, setRatings] = useState({
        answer_rating: 5,
        answer_comment: '',
        chunks_rating: 5,
        chunks_comment: '',
        articles_rating: 5,
        articles_comment: '',
        confirm_action: true
    });

    const handleSubmit = async (e) => {
        e.preventDefault();
        try {
            await evaluationService.submitEvaluation({
                task_id: task.id,
                ...ratings
            });
            alert('Evaluation submitted successfully!');
            navigate('/reviewer/eval');
        } catch (err) {
            alert('Failed to submit evaluation.');
        }
    };

    if (!task) return <div className="error-box">Task not found.</div>;

    const isFormValid = ratings.answer_rating !== null && ratings.chunks_rating !== null && ratings.articles_rating !== null;

    return (
        <div className="evaluation-form-container">
            <header className="form-header">
                <h1>Evaluation Task: {task.external_id}</h1>
                <button className="back-btn" onClick={() => navigate('/reviewer/eval')}>&larr; Back to Tasks</button>
            </header>

            <div className="form-grid">
                {/* Read-Only Context Section */}
                <section className="read-only-section">
                    <div className="spec-card">
                        <h3>Question</h3>
                        <p>{task.question}</p>
                    </div>
                    <div className="spec-card highlight">
                        <h3>AI Answer</h3>
                        <p>{task.answer}</p>
                    </div>
                    <div className="spec-card">
                        <h3>Retrieved Chunks</h3>
                        <div className="chunks-list">
                            {task.chunks.map((c, i) => (
                                <div key={i} className="chunk-item">
                                    <span className="idx">{i + 1}</span>
                                    <p>{c}</p>
                                </div>
                            ))}
                        </div>
                    </div>
                    <div className="spec-card">
                        <h3>Cited Articles</h3>
                        <div className="tags">
                            {task.articles.map((a, i) => <span key={i} className="tag">{a}</span>)}
                        </div>
                    </div>
                </section>

                {/* Interactive Rating Section */}
                <section className="rating-section">
                    <form onSubmit={handleSubmit} className="spec-form">
                        <div className="rating-block">
                            <div className="rating-header">
                                <h3>Answer Quality</h3>
                                <div className="score-display">{ratings.answer_rating}/10</div>
                            </div>
                            <input type="range" min="0" max="10" step="1"
                                value={ratings.answer_rating}
                                onChange={(e) => setRatings({ ...ratings, answer_rating: parseInt(e.target.value) })} />
                            <textarea
                                placeholder="What could be improved in the answer?"
                                value={ratings.answer_comment}
                                onChange={(e) => setRatings({ ...ratings, answer_comment: e.target.value })}
                            />
                        </div>

                        <div className="rating-block">
                            <div className="rating-header">
                                <h3>Chunks Relevance</h3>
                                <div className="score-display">{ratings.chunks_rating}/10</div>
                            </div>
                            <input type="range" min="0" max="10" step="1"
                                value={ratings.chunks_rating}
                                onChange={(e) => setRatings({ ...ratings, chunks_rating: parseInt(e.target.value) })} />
                            <textarea
                                placeholder="Are the retrieved chunks relevant to the question?"
                                value={ratings.chunks_comment}
                                onChange={(e) => setRatings({ ...ratings, chunks_comment: e.target.value })}
                            />
                        </div>

                        <div className="rating-block">
                            <div className="rating-header">
                                <h3>Articles Accuracy</h3>
                                <div className="score-display">{ratings.articles_rating}/10</div>
                            </div>
                            <input type="range" min="0" max="10" step="1"
                                value={ratings.articles_rating}
                                onChange={(e) => setRatings({ ...ratings, articles_rating: parseInt(e.target.value) })} />
                            <textarea
                                placeholder="Are the referenced articles correct?"
                                value={ratings.articles_comment}
                                onChange={(e) => setRatings({ ...ratings, articles_comment: e.target.value })}
                            />
                        </div>

                        <div className="decision-block">
                            <h3>Final Decision</h3>
                            <div className="radio-group">
                                <label className="check-label">
                                    <input type="radio" name="decision" checked={ratings.confirm_action === true}
                                        onChange={() => setRatings({ ...ratings, confirm_action: true })} />
                                    Confirm (Keep)
                                </label>
                                <label className="check-label">
                                    <input type="radio" name="decision" checked={ratings.confirm_action === false}
                                        onChange={() => setRatings({ ...ratings, confirm_action: false })} />
                                    Change Required
                                </label>
                            </div>
                        </div>

                        <button type="submit" className="submit-btn" disabled={!isFormValid}>
                            Submit Evaluation
                        </button>
                    </form>
                </section>
            </div>
        </div>
    );
};

export default EvaluationForm;
