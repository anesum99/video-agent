#!/bin/bash
# deploy-test.sh - Deploy test version of video agent system

set -e

# Build test version
docker-compose -f docker-compose-test.yml down
DOCKER_BUILDKIT=0 docker-compose -f docker-compose-test.yml build

# Start test version
docker-compose -f docker-compose-test.yml up -d

echo "⏳ Waiting for services to start..."
sleep 30

# Check test version status
echo "📊 Test version status:"
docker-compose -f docker-compose-test.yml ps

echo ""
echo "🌐 Access points:"
echo "  Stable version: http://18.191.231.172 (port 80)"
echo "  Test version:   http://18.191.231.172:8080 (port 8080)"
echo ""
echo "💾 Data separation:"
echo "  Stable uploads: ./uploads/"
echo "  Test uploads:   ./uploads-test/"
echo ""
echo "🔧 Management commands:"
echo "  View test logs:  docker-compose -f docker-compose-test.yml logs -f"
echo "  Stop test:       docker-compose -f docker-compose-test.yml down"
echo "  Stop stable:     docker-compose -f docker-compose-microservices.yml down"
echo ""
echo "✅ Test deployment complete!"
