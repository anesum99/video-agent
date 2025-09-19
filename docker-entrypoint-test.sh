#!/bin/bash

# Docker entrypoint script with working IP discovery
set -e

echo "🚀 Starting Video Agent..."

# Create necessary directories
mkdir -p /app/logs
mkdir -p /app/uploads
mkdir -p /app/processed

# Wait for HTTP microservices to be available
echo "⏳ Waiting for microservices..."
sleep 15

# Function to get IP using getent (more reliable than nslookup in containers)
get_service_ip() {
    local service_name=$1
    
    # Try getent first
    local ip=$(getent hosts $service_name 2>/dev/null | awk '{print $1}' || echo "")
    
    if [ -n "$ip" ] && [ "$ip" != "" ]; then
        echo "$ip"
        return 0
    fi
    
    # Fallback: try to extract IP from successful curl connection
    # Since curl works, we can extract the IP from a successful connection
    local ip=$(timeout 5 curl -s --connect-timeout 2 "http://$service_name:8011/health" \
               --trace-ascii /tmp/trace_$service_name 2>/dev/null \
               && grep "Connected to" /tmp/trace_$service_name | awk '{print $3}' | tr -d '()' || echo "")
    
    if [ -n "$ip" ] && [ "$ip" != "" ]; then
        echo "$ip"
        return 0
    fi
    
    # If all else fails, return the service name
    echo "$service_name"
}

# Discover actual service IPs
echo "🔍 Discovering service IPs..."

declare -A service_ips

services=("router-test" "yolo-test" "whisper-test" "ocr-test" "scenes-test" "ffmpeg-test")
for service in "${services[@]}"; do
    echo "Resolving $service..."
    
    ip=$(get_service_ip $service)
    service_ips[$service]=$ip
    
    if [ "$ip" != "$service" ]; then
        echo "✅ Found $service at $ip"
    else
        echo "⚠️ Using hostname for $service"
    fi
done

# Set environment variables with discovered IPs
echo "📝 Setting service URLs with actual IPs..."
export ROUTER_URL="http://${service_ips[router-test]}:8016"
export YOLO_URL="http://${service_ips[yolo-test]}:8011"
export WHISPER_URL="http://${service_ips[whisper-test]}:8012"
export OCR_URL="http://${service_ips[ocr-test]}:8013"
export SCENES_URL="http://${service_ips[scenes-test]}:8014"
export FFMPEG_URL="http://${service_ips[ffmpeg-test]}:8015"

# Test connectivity to services
echo "🔍 Testing microservice connectivity..."

for service in "${services[@]}"; do
    port=""
    case $service in
        router-test) port="8016" ;;
        yolo-test) port="8011" ;;
        whisper-test) port="8012" ;;
        ocr-test) port="8013" ;;
        scenes-test) port="8014" ;;
        ffmpeg-test) port="8015" ;;
    esac
    
    ip=${service_ips[$service]}
    
    if curl -s --connect-timeout 5 "http://$ip:$port/health" > /dev/null 2>&1; then
        echo "✅ $service service available at $ip:$port"
    else
        echo "⚠️ $service service not available - will fallback to Gemini-only"
    fi
done

# Print final URLs for debugging
echo "🌐 Final Service URLs:"
echo "  Router: $ROUTER_URL"
echo "  YOLO: $YOLO_URL"
echo "  Whisper: $WHISPER_URL"
echo "  OCR: $OCR_URL"
echo "  Scenes: $SCENES_URL"
echo "  FFmpeg: $FFMPEG_URL"

# Start the main Flask application
echo "🌟 Starting main application with discovered IPs..."
exec gunicorn --worker-class eventlet -w 1 --bind 0.0.0.0:5001 --timeout 120 app:app
