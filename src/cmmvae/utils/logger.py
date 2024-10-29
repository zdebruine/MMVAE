# import logging
# import os

# def setup_logging(log_file=None, level=logging.INFO):
#     print("Setting up logging")
#     # Clear existing handlers
#     logging.root.handlers = []
#     logging.root.setLevel(level)

#     formatter = logging.Formatter(
#         '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
#     )

#     rank = int(os.environ.get('LOCAL_RANK', 0))

#     if rank == 0:
#         # Main process logs to file and console
#         handlers = []

#         if log_file:
#             fh = logging.FileHandler(log_file)
#             fh.setFormatter(formatter)
#             handlers.append(fh)

#         ch = logging.StreamHandler()
#         ch.setFormatter(formatter)
#         handlers.append(ch)

#         for handler in handlers:
#             logging.root.addHandler(handler)
#     else:
#         # Other processes discard logs
#         logging.root.addHandler(logging.NullHandler())
