# gunicorn.conf.py  — memory-optimised for classification tasks

import os

# Binding
bind = f"0.0.0.0:{os.environ.get('PORT', 8080)}"

# Workers: single worker to reduce memory footprint (each worker loads a full classifier)
workers = 1

# Use gthread to handle concurrency without forking
worker_class = 'gthread'
threads = 4

# Timeouts — AI calls can take 60-120s for large batches
timeout = 300
graceful_timeout = 120
keepalive = 5

# Logging
accesslog = '-'
errorlog = '-'
loglevel = 'info'
access_log_format = '%(h)s "%(r)s" %(s)s %(b)s %(D)sµs'

# Prevent memory bloat on long-running workers
max_requests = 200
max_requests_jitter = 50