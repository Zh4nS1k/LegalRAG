import axios from 'axios';

export function setupLatencyTracking() {
  // 1. Fetch Interceptor (used by ChatSection.js)
  const originalFetch = window.fetch;
  window.fetch = async function(...args) {
    const [url, config] = args;
    const isApi = typeof url === 'string' && url.includes('/api/');
    
    if (!isApi) {
      return originalFetch(...args);
    }

    const traceId = `trace_${Date.now()}_${Math.random().toString(36).substr(2, 5)}`;
    const start = performance.now(); // Request Start
    
    // Inject X-Trace-ID
    const newConfig = { 
      ...config, 
      headers: { ...config?.headers, 'X-Trace-ID': traceId } 
    };
    
    const response = await originalFetch(url, newConfig);
    const firstByte = performance.now(); // First Byte Received
    
    // Intercept JSON parsing to calculate "Response Fully Parsed"
    const originalJson = response.json.bind(response);
    response.json = async function() {
      const data = await originalJson();
      const end = performance.now(); // Response Fully Parsed
      
      if (data && data.trace_report) {
        const { go_processing, python_rag_total } = data.trace_report.metrics_ms || {};
        const backendTotal = (go_processing || 0) + (python_rag_total || 0);
        
        // Network RTT estimation
        data.trace_report.metrics_ms.network_rtt = Math.max(0, Math.round((end - start) - backendTotal));
        
        // Output "Performance Manifest"
        console.log('🚀 Performance Manifest [Fetch]:\n', JSON.stringify({ trace_report: data.trace_report }, null, 2));
        console.log(`React Timings: Time-to-First-Byte=${(firstByte-start).toFixed(2)}ms, Parsing=${(end-firstByte).toFixed(2)}ms`);
      }
      return data;
    };
    
    return response;
  };

  // 2. Axios Interceptor (used by evaluationService.js)
  axios.interceptors.request.use((config) => {
    config.headers['X-Trace-ID'] = `trace_${Date.now()}_${Math.random().toString(36).substr(2, 5)}`;
    config.metadata = { startTime: performance.now() };
    return config;
  });

  axios.interceptors.response.use((response) => {
    const end = performance.now();
    const start = response.config.metadata.startTime;
    
    if (response.data && response.data.trace_report) {
       const { go_processing, python_rag_total } = response.data.trace_report.metrics_ms || {};
       const backendTotal = (go_processing || 0) + (python_rag_total || 0);
       response.data.trace_report.metrics_ms.network_rtt = Math.max(0, Math.round((end - start) - backendTotal));
       
       console.log('🚀 Performance Manifest [Axios]:\n', JSON.stringify({ trace_report: response.data.trace_report }, null, 2));
    }
    return response;
  });
}
