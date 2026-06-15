import os
import sys
from unittest.mock import patch, MagicMock

# Mock connectivity to return True for internet
# Mock pinecone to raise NameResolutionError

def test_startup_with_dns_failure():
    # Set PYTHONPATH
    sys.path.append(os.getcwd())
    
    from ai_service.lifecycle_hooks import pre_flight_check
    
    with patch("ai_service.utils.connectivity.is_internet_available", return_value=True), \
         patch("pinecone.Pinecone") as mock_pinecone:
        
        # Simulate NameResolutionError
        mock_instance = mock_pinecone.return_value
        mock_instance.list_indexes.side_effect = Exception("HTTPSConnection(host='api.pinecone.io', port=443): Failed to resolve 'api.pinecone.io' ([Errno -3] Temporary failure in name resolution)")
        
        print("Running pre_flight_check with simulated DNS failure...")
        # Should NOT exit
        pre_flight_check()
        print("pre_flight_check finished successfully (without exiting).")
        
        # Check if environment variable was set
        assert os.environ.get("LEGAL_RAG_DISABLE_PINECONE") == "1"
        print("Verified: LEGAL_RAG_DISABLE_PINECONE is set to 1")

if __name__ == "__main__":
    test_startup_with_dns_failure()
