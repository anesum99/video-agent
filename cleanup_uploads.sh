#!/bin/bash
cd /home/ec2-user/video-agent/uploads
BEFORE=$(du -sh . | cut -f1)
ls -t *.mp4 2>/dev/null | tail -n +4 | xargs -r rm -f
AFTER=$(du -sh . | cut -f1)
echo "$(date): Cleaned uploads - was $BEFORE, now $AFTER, kept 3 newest"
