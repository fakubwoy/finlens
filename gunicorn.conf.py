# gunicorn.conf.py  — tuned for long-running AI classification requests

import os

# Binding
bind    = f"0.0.0.0:{os.environ.get('PORT', 8080)}"

# Workers
workers     = int(os.environ.get('WEB_CONCURRENCY', 2))
worker_class = 'gthread'   # threaded worker handles blocking AI calls without timing out
threads     = 4             # threads per worker

# Timeouts — AI calls can take 60-120s for large batches
timeout      = 300          # 5 min hard kill
graceful_timeout = 120      # 2 min to finish in-flight request on SIGTERM
keepalive    = 5

# Logging
accesslog  = '-'
errorlog   = '-'
loglevel   = 'info'
access_log_format = '%(h)s "%(r)s" %(s)s %(b)s %(D)sµs'

# Prevent memory bloat on long-running workers
max_requests         = 500
max_requests_jitter  = 50