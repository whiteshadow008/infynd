// frontend/script.js

// 1. Configuration Constants
const API_BASE_URL = "http://127.0.0.1:5000"; // Ensure this matches your Flask port
const SIMULATED_USER_EMAIL = "preethavijagan@gmail.com"; // Get this dynamically in a real app

// 2. DOM Element Selection
const chartSegments = document.querySelectorAll('.chart-placeholder .segment');
const personalizedContentArea = document.getElementById('personalized-content-area');
const simulatedUserIdElement = document.getElementById('simulated-user-id');

// 3. Event Listeners
document.addEventListener('DOMContentLoaded', () => {
    // You could dynamically extract the user email from the DOM if it changes
    // const currentUserEmail = simulatedUserIdElement.textContent.split(': ')[1].trim();
    const currentUserEmail = SIMULATED_USER_EMAIL; // Using the constant for this demo

    chartSegments.forEach(segment => {
        segment.addEventListener('click', async () => {
            const domain = segment.dataset.domain;
            console.log(`User clicked on domain: ${domain}`);

            // Update UI to show loading state
            personalizedContentArea.innerHTML = `<p>Generating personalized content for <strong>${domain}</strong>...</p>`;

            // 4. API Call Logic (async function)
            try {
                const response = await fetch(`${API_BASE_URL}/get-personalized-content`, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ domain: domain, user_email: currentUserEmail })
                });

                if (!response.ok) {
                    // If the response is not OK (e.g., 400, 500 status)
                    const errorData = await response.json();
                    throw new Error(errorData.error || `HTTP error! Status: ${response.status}`);
                }

                const data = await response.json();
                console.log("Backend response:", data);

                // 5. Update UI Based on API Response
                if (data.success && data.personalized_content) {
                    const content = data.personalized_content;
                    personalizedContentArea.innerHTML = `
                        <div class="personalized-item">
                            <div class="rec-label">#1 RECOMMENDED</div>
                            <div class="blog-type">AI-Generated Blog - ${content.domain}</div>
                            <h3>${content.title}</h3>
                            <p>${content.short_description}</p>
                            <div class="match-score">Match Score: ${content.match_score}</div>
                        </div>
                        <p class="email-status">Email Status: ${content.email_status === 'sent' ? 'Sent successfully!' : 'Failed to send.'}</p>
                    `;
                    alert(`New content for "${content.title}" generated and an email has been ${content.email_status === 'sent' ? 'sent' : 'attempted to send'} to ${currentUserEmail}!`);
                } else {
                    personalizedContentArea.innerHTML = `<p>Error generating content: ${data.error || 'Unknown error'}</p>`;
                    alert(`Error: ${data.error || 'Unknown error'}`);
                }

            } catch (error) {
                console.error('Fetch error:', error);
                personalizedContentArea.innerHTML = `<p>Failed to load personalized content. Please check the backend server and your network connection.<br>Error: ${error.message}</p>`;
                alert(`Failed to perform action: ${error.message}`);
            }
        });
    });
});
// --- ENGAGEMENT DASHBOARD VISUALIZATION (Enhancement) ---
        async function updateEngagementDashboard(userId) {
            // FIX: Ensure API returns data and chart updates correctly
            try {
                const response = await fetch(`${ENGAGEMENT_URL}?userId=${userId}`);
                if (!response.ok) throw new Error('Failed to fetch engagement data');
                const data = await response.json();
                
                renderChart(data.labels, data.counts);
            } catch (error) {
                console.error('Error fetching engagement data:', error);
            }
        }

        function renderChart(labels, data) {
            const ctx = document.getElementById('engagementChart').getContext('2d');
            
            if (engagementChartInstance) {
                engagementChartInstance.destroy();
            }

            engagementChartInstance = new Chart(ctx, {
                type: 'doughnut',
                data: {
                    labels: labels,
                    datasets: [{
                        label: 'Interactions by Topic',
                        data: data,
                        backgroundColor: [
                            'rgba(0, 123, 255, 0.7)', 
                            'rgba(40, 167, 69, 0.7)', 
                            'rgba(255, 193, 7, 0.7)', 
                            'rgba(23, 162, 184, 0.7)', 
                            'rgba(108, 117, 125, 0.7)', 
                        ],
                        hoverOffset: 10
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: { position: 'bottom', labels: { boxWidth: 10 } },
                        title: { display: false }
                    }
                }
            });
        }
        
        // --- INITIALIZATION ---
        document.addEventListener('DOMContentLoaded', () => {
            // Initial track for PAGE_VIEW
            trackEvent('PAGE_VIEW', 'homepage', 'general'); 
            
            // Initial call to populate recommendations and dashboard on load
            requestRecommendations();
        });


// 6. Any other client-side functions can go here