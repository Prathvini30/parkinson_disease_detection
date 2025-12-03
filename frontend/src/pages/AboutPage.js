import React from 'react';
import './AboutPage.css';

const AboutPage = () => {
  return (
    <div className="about-page">
      <div className="about-image-container">
        <img src="https://images.unsplash.com/photo-1550831107-1553da8c8464?q=80&w=1887&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D" alt="Healthcare professional" />
        <img src="https://images.unsplash.com/photo-1579684385127-6c2a895f3c3d?q=80&w=1887&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D" alt="Doctor with patient" />
        <img src="https://images.unsplash.com/photo-1584820927239-433d362a78df?q=80&w=1887&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D" alt="Medical research" />
      </div>
      <div className="about-content-container">
        <div className="about-header">
          <h1>Parkinson's Disease</h1>
        </div>
        <div className="about-text">
          <div className="about-section">
            <h2>What is Parkinson's Disease?</h2>
            <p>
              Parkinson's disease is a progressive disorder that affects the nervous system and the parts of the body controlled by the nerves. Symptoms start slowly. The first symptom may be a barely noticeable tremor in just one hand. Tremors are common, but the disorder also commonly causes stiffness or slowing of movement.
            </p>
          </div>
          <div className="about-section">
            <h2>History</h2>
            <p>
            The history of Parkinson's disease is a fascinating journey of medical discovery. While symptoms of the disease have been described in ancient texts, it was in 1817 that James Parkinson, an English surgeon, first published a detailed medical essay on the condition. His work, "An Essay on the Shaking Palsy," provided the first clear and comprehensive description of the symptoms, which he called "paralysis agitans" or shaking palsy. Parkinson's meticulous observations laid the foundation for future research.

For many decades, little progress was made in understanding the underlying cause of the disease. However, in the 1960s, a breakthrough occurred when researchers identified that the symptoms of Parkinson's were linked to the depletion of dopamine, a neurotransmitter in the brain. This discovery revolutionized the treatment of the disease, leading to the development of levodopa, a medication that remains the most effective treatment for Parkinson's symptoms today.

Over the years, our understanding of Parkinson's disease has continued to evolve. Researchers have identified genetic and environmental factors that may contribute to the disease, and ongoing studies are exploring new treatments, including deep brain stimulation (DBS) and other innovative therapies. The history of Parkinson's disease is a testament to the power of medical research and the ongoing quest to improve the lives of those affected by this challenging condition.
            </p>
          </div>
          <div className="about-section">
            <h2>Symptoms</h2>
            <p>
              Parkinson's disease signs and symptoms can be different for everyone. Early signs may be mild and go unnoticed. Symptoms often begin on one side of your body and usually remain worse on that side, even after symptoms begin to affect both sides.
            </p>
            <ul>
              <li><strong>Tremor.</strong> A tremor, or shaking, usually begins in a limb, often your hand or fingers.</li>
              <li><strong>Slowed movement (bradykinesia).</strong> Over time, Parkinson's disease may slow your movement, making simple tasks difficult and time-consuming.</li>
              <li><strong>Rigid muscles.</strong> Muscle stiffness may occur in any part of your body. The stiff muscles can be painful and limit your range of motion.</li>
              <li><strong>Impaired posture and balance.</strong> Your posture may become stooped, or you may have balance problems as a result of Parkinson's disease.</li>
              <li><strong>Loss of automatic movements.</strong> You may have a decreased ability to perform unconscious movements, including blinking, smiling or swinging your arms when you walk.</li>
              <li><strong>Speech changes.</strong> You may speak softly, quickly, slur or hesitate before talking. Your speech may be more of a monotone rather than with the usual inflections.</li>
              <li><strong>Writing changes.</strong> It may become hard to write, and your writing may appear small.</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
};

export default AboutPage;
