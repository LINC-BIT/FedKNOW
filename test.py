import flwr as fl
from NewFedAvg import OurFed
strategy = OurFed(
    min_fit_clients=1,
    min_eval_clients=1,
    fraction_eval =1,
    fraction_fit=0.5,
    min_available_clients = 1
)
fl.server.start_server(server_address='10.1.114.64:8000',config={"num_rounds": 99},strategy=strategy)
# ssh://omnisky@10.1.114.64:22/home/omnisky/anaconda3/envs/lpyx_pytorch/bin/python -u /data/lpyx/FedFPKD/test.py
# INFO flower 2022-04-23 18:16:39,259 | app.py:80 | Flower server running (insecure, 99 rounds)
# INFO flower 2022-04-23 18:16:39,259 | server.py:118 | Initializing global parameters
# INFO flower 2022-04-23 18:16:39,259 | server.py:304 | Requesting initial parameters from one random client
# INFO flower 2022-04-23 18:17:02,402 | server.py:307 | Received initial parameters from one random client
# INFO flower 2022-04-23 18:17:02,402 | server.py:120 | Evaluating initial parameters
# INFO flower 2022-04-23 18:17:02,402 | server.py:133 | FL starting
# DEBUG flower 2022-04-23 18:17:35,512 | server.py:255 | fit_round: strategy sampled 5 clients (out of 10)
# DEBUG flower 2022-04-23 18:19:59,071 | server.py:264 | fit_round received 5 results and 0 failures
# DEBUG flower 2022-04-23 18:19:59,565 | server.py:205 | evaluate_round: strategy sampled 10 clients (out of 10)
# DEBUG flower 2022-04-23 18:20:12,014 | server.py:214 | evaluate_round received 10 results and 0 failures
# DEBUG flower 2022-04-23 18:20:12,015 | server.py:255 | fit_round: strategy sampled 5 clients (out of 10)
# DEBUG flower 2022-04-23 18:22:35,012 | server.py:264 | fit_round received 5 results and 0 failures
# DEBUG flower 2022-04-23 18:22:35,472 | server.py:205 | evaluate_round: strategy sampled 10 clients (out of 10)
# DEBUG flower 2022-04-23 18:22:44,355 | server.py:214 | evaluate_round received 10 results and 0 failures
# DEBUG flower 2022-04-23 18:22:44,356 | server.py:255 | fit_round: strategy sampled 5 clients (out of 10)
# DEBUG flower 2022-04-23 18:24:18,641 | server.py:264 | fit_round received 5 results and 0 failures
# DEBUG flower 2022-04-23 18:24:19,108 | server.py:205 | evaluate_round: strategy sampled 10 clients (out of 10)
# DEBUG flower 2022-04-23 18:24:28,324 | server.py:214 | evaluate_round received 10 results and 0 failures
# DEBUG flower 2022-04-23 18:24:28,325 | server.py:255 | fit_round: strategy sampled 5 clients (out of 10)
# DEBUG flower 2022-04-23 18:25:32,815 | server.py:264 | fit_round received 5 results and 0 failures
# DEBUG flower 2022-04-23 18:25:33,229 | server.py:205 | evaluate_round: strategy sampled 10 clients (out of 10)
# DEBUG flower 2022-04-23 18:25:42,780 | server.py:214 | evaluate_round received 10 results and 0 failures
# DEBUG flower 2022-04-23 18:25:42,781 | server.py:255 | fit_round: strategy sampled 5 clients (out of 10)
# DEBUG flower 2022-04-23 18:26:51,834 | server.py:264 | fit_round received 5 results and 0 failures
# DEBUG flower 2022-04-23 18:26:52,220 | server.py:205 | evaluate_round: strategy sampled 10 clients (out of 10)
# DEBUG flower 2022-04-23 18:26:59,937 | server.py:214 | evaluate_round received 10 results and 0 failures
# DEBUG flower 2022-04-23 18:26:59,938 | server.py:255 | fit_round: strategy sampled 5 clients (out of 10)
# DEBUG flower 2022-04-23 18:28:08,652 | server.py:264 | fit_round received 5 results and 0 failures
# DEBUG flower 2022-04-23 18:28:09,083 | server.py:205 | evaluate_round: strategy sampled 10 clients (out of 10)
# DEBUG flower 2022-04-23 18:28:17,318 | server.py:214 | evaluate_round received 10 results and 0 failures
# DEBUG flower 2022-04-23 18:28:17,318 | server.py:255 | fit_round: strategy sampled 5 clients (out of 10)
# DEBUG flower 2022-04-23 18:29:49,879 | server.py:264 | fit_round received 5 results and 0 failures
# DEBUG flower 2022-04-23 18:29:50,298 | server.py:205 | evaluate_round: strategy sampled 10 clients (out of 10)
# DEBUG flower 2022-04-23 18:30:00,026 | server.py:214 | evaluate_round received 10 results and 0 failures
# DEBUG flower 2022-04-23 18:30:00,027 | server.py:255 | fit_round: strategy sampled 5 clients (out of 10)
# DEBUG flower 2022-04-23 18:32:24,865 | server.py:264 | fit_round received 5 results and 0 failures
# DEBUG flower 2022-04-23 18:32:25,265 | server.py:205 | evaluate_round: strategy sampled 10 clients (out of 10)
# DEBUG flower 2022-04-23 18:32:34,170 | server.py:214 | evaluate_round received 10 results and 0 failures
# DEBUG flower 2022-04-23 18:32:34,171 | server.py:255 | fit_round: strategy sampled 5 clients (out of 10)
# DEBUG flower 2022-04-23 18:33:42,337 | server.py:264 | fit_round received 5 results and 0 failures
# DEBUG flower 2022-04-23 18:33:42,750 | server.py:205 | evaluate_round: strategy sampled 10 clients (out of 10)
# DEBUG flower 2022-04-23 18:33:52,977 | server.py:214 | evaluate_round received 10 results and 0 failures
# DEBUG flower 2022-04-23 18:33:52,978 | server.py:255 | fit_round: strategy sampled 5 clients (out of 10)
# DEBUG flower 2022-04-23 18:35:39,237 | server.py:264 | fit_round received 5 results and 0 failures
# DEBUG flower 2022-04-23 18:35:39,629 | server.py:205 | evaluate_round: strategy sampled 10 clients (out of 10)
# DEBUG flower 2022-04-23 18:35:51,655 | server.py:214 | evaluate_round received 10 results and 0 failures
# DEBUG flower 2022-04-23 18:35:51,656 | server.py:255 | fit_round: strategy sampled 5 clients (out of 10)
# DEBUG flower 2022-04-23 18:37:57,159 | server.py:264 | fit_round received 5 results and 0 failures


# DEBUG flower 2022-04-23 18:37:57,540 | server.py:205 | evaluate_round: strategy sampled 10 clients (out of 10)
# DEBUG flower 2022-04-23 18:38:08,550 | server.py:214 | evaluate_round received 10 results and 0 failures
# DEBUG flower 2022-04-23 18:38:08,551 | server.py:255 | fit_round: strategy sampled 5 clients (out of 10)
# DEBUG flower 2022-04-23 18:40:08,375 | server.py:264 | fit_round received 5 results and 0 failures
# DEBUG flower 2022-04-23 18:40:08,729 | server.py:205 | evaluate_round: strategy sampled 10 clients (out of 10)
# DEBUG flower 2022-04-23 18:40:20,726 | server.py:214 | evaluate_round received 10 results and 0 failures
# DEBUG flower 2022-04-23 18:40:20,726 | server.py:255 | fit_round: strategy sampled 5 clients (out of 10)
# DEBUG flower 2022-04-23 18:44:36,326 | server.py:264 | fit_round received 5 results and 0 failures
# DEBUG flower 2022-04-23 18:44:36,695 | server.py:205 | evaluate_round: strategy sampled 10 clients (out of 10)
# DEBUG flower 2022-04-23 18:44:48,042 | server.py:214 | evaluate_round received 10 results and 0 failures
# DEBUG flower 2022-04-23 18:44:48,043 | server.py:255 | fit_round: strategy sampled 5 clients (out of 10)
# DEBUG flower 2022-04-23 18:47:37,487 | server.py:264 | fit_round received 5 results and 0 failures
# DEBUG flower 2022-04-23 18:47:37,883 | server.py:205 | evaluate_round: strategy sampled 10 clients (out of 10)
# DEBUG flower 2022-04-23 18:47:49,911 | server.py:214 | evaluate_round received 10 results and 0 failures
# DEBUG flower 2022-04-23 18:47:49,911 | server.py:255 | fit_round: strategy sampled 5 clients (out of 10)
# DEBUG flower 2022-04-23 18:52:02,513 | server.py:264 | fit_round received 5 results and 0 failures
# DEBUG flower 2022-04-23 18:52:02,863 | server.py:205 | evaluate_round: strategy sampled 10 clients (out of 10)
# DEBUG flower 2022-04-23 18:52:14,103 | server.py:214 | evaluate_round received 10 results and 0 failures
# DEBUG flower 2022-04-23 18:52:14,104 | server.py:255 | fit_round: strategy sampled 5 clients (out of 10)
# DEBUG flower 2022-04-23 18:54:15,450 | server.py:264 | fit_round received 5 results and 0 failures
# DEBUG flower 2022-04-23 18:54:15,828 | server.py:205 | evaluate_round: strategy sampled 10 clients (out of 10)
# DEBUG flower 2022-04-23 18:54:27,270 | server.py:214 | evaluate_round received 10 results and 0 failures
# DEBUG flower 2022-04-23 18:54:27,270 | server.py:255 | fit_round: strategy sampled 5 clients (out of 10)
# DEBUG flower 2022-04-23 18:56:27,189 | server.py:264 | fit_round received 5 results and 0 failures
# DEBUG flower 2022-04-23 18:56:27,571 | server.py:205 | evaluate_round: strategy sampled 10 clients (out of 10)
# DEBUG flower 2022-04-23 18:56:39,743 | server.py:214 | evaluate_round received 10 results and 0 failures
# DEBUG flower 2022-04-23 18:56:39,744 | server.py:255 | fit_round: strategy sampled 5 clients (out of 10)
# DEBUG flower 2022-04-23 19:00:54,283 | server.py:264 | fit_round received 5 results and 0 failures
# DEBUG flower 2022-04-23 19:00:54,702 | server.py:205 | evaluate_round: strategy sampled 10 clients (out of 10)
# DEBUG flower 2022-04-23 19:01:06,314 | server.py:214 | evaluate_round received 10 results and 0 failures
# DEBUG flower 2022-04-23 19:01:06,314 | server.py:255 | fit_round: strategy sampled 5 clients (out of 10)
# DEBUG flower 2022-04-23 19:03:55,614 | server.py:264 | fit_round received 5 results and 0 failures
# DEBUG flower 2022-04-23 19:03:55,980 | server.py:205 | evaluate_round: strategy sampled 10 clients (out of 10)
# DEBUG flower 2022-04-23 19:04:07,462 | server.py:214 | evaluate_round received 10 results and 0 failures
# DEBUG flower 2022-04-23 19:04:07,463 | server.py:255 | fit_round: strategy sampled 5 clients (out of 10)
# DEBUG flower 2022-04-23 19:07:27,148 | server.py:264 | fit_round received 5 results and 0 failures
# DEBUG flower 2022-04-23 19:07:27,504 | server.py:205 | evaluate_round: strategy sampled 10 clients (out of 10)
# DEBUG flower 2022-04-23 19:07:41,986 | server.py:214 | evaluate_round received 10 results and 0 failures
# DEBUG flower 2022-04-23 19:07:41,987 | server.py:255 | fit_round: strategy sampled 5 clients (out of 10)
# DEBUG flower 2022-04-23 19:11:00,891 | server.py:264 | fit_round received 5 results and 0 failures


# DEBUG flower 2022-04-23 19:11:01,279 | server.py:205 | evaluate_round: strategy sampled 10 clients (out of 10)
# DEBUG flower 2022-04-23 19:11:17,527 | server.py:214 | evaluate_round received 10 results and 0 failures
# DEBUG flower 2022-04-23 19:11:17,528 | server.py:255 | fit_round: strategy sampled 5 clients (out of 10)
# DEBUG flower 2022-04-23 19:13:30,895 | server.py:264 | fit_round received 5 results and 0 failures
# DEBUG flower 2022-04-23 19:13:31,272 | server.py:205 | evaluate_round: strategy sampled 10 clients (out of 10)
# DEBUG flower 2022-04-23 19:13:45,011 | server.py:214 | evaluate_round received 10 results and 0 failures
# DEBUG flower 2022-04-23 19:13:45,011 | server.py:255 | fit_round: strategy sampled 5 clients (out of 10)
# DEBUG flower 2022-04-23 19:15:59,677 | server.py:264 | fit_round received 5 results and 0 failures
# DEBUG flower 2022-04-23 19:16:00,014 | server.py:205 | evaluate_round: strategy sampled 10 clients (out of 10)
# DEBUG flower 2022-04-23 19:16:14,992 | server.py:214 | evaluate_round received 10 results and 0 failures
# DEBUG flower 2022-04-23 19:16:14,993 | server.py:255 | fit_round: strategy sampled 5 clients (out of 10)
# DEBUG flower 2022-04-23 19:18:38,864 | server.py:264 | fit_round received 5 results and 0 failures
# DEBUG flower 2022-04-23 19:18:39,276 | server.py:205 | evaluate_round: strategy sampled 10 clients (out of 10)
# DEBUG flower 2022-04-23 19:18:54,499 | server.py:214 | evaluate_round received 10 results and 0 failures
# DEBUG flower 2022-04-23 19:18:54,500 | server.py:255 | fit_round: strategy sampled 5 clients (out of 10)
# DEBUG flower 2022-04-23 19:21:21,195 | server.py:264 | fit_round received 5 results and 0 failures
# DEBUG flower 2022-04-23 19:21:21,651 | server.py:205 | evaluate_round: strategy sampled 10 clients (out of 10)
# DEBUG flower 2022-04-23 19:21:38,307 | server.py:214 | evaluate_round received 10 results and 0 failures
# DEBUG flower 2022-04-23 19:21:38,308 | server.py:255 | fit_round: strategy sampled 5 clients (out of 10)
# DEBUG flower 2022-04-23 19:23:53,943 | server.py:264 | fit_round received 5 results and 0 failures
# DEBUG flower 2022-04-23 19:23:54,325 | server.py:205 | evaluate_round: strategy sampled 10 clients (out of 10)
# DEBUG flower 2022-04-23 19:24:09,003 | server.py:214 | evaluate_round received 10 results and 0 failures
# DEBUG flower 2022-04-23 19:24:09,004 | server.py:255 | fit_round: strategy sampled 5 clients (out of 10)
# DEBUG flower 2022-04-23 19:27:29,661 | server.py:264 | fit_round received 5 results and 0 failures
# DEBUG flower 2022-04-23 19:27:30,043 | server.py:205 | evaluate_round: strategy sampled 10 clients (out of 10)
# DEBUG flower 2022-04-23 19:27:44,819 | server.py:214 | evaluate_round received 10 results and 0 failures
# DEBUG flower 2022-04-23 19:27:44,820 | server.py:255 | fit_round: strategy sampled 5 clients (out of 10)
# DEBUG flower 2022-04-23 19:31:03,955 | server.py:264 | fit_round received 5 results and 0 failures
# DEBUG flower 2022-04-23 19:31:04,323 | server.py:205 | evaluate_round: strategy sampled 10 clients (out of 10)
# DEBUG flower 2022-04-23 19:31:20,152 | server.py:214 | evaluate_round received 10 results and 0 failures
# DEBUG flower 2022-04-23 19:31:20,152 | server.py:255 | fit_round: strategy sampled 5 clients (out of 10)
# DEBUG flower 2022-04-23 19:33:40,120 | server.py:264 | fit_round received 5 results and 0 failures
# DEBUG flower 2022-04-23 19:33:40,515 | server.py:205 | evaluate_round: strategy sampled 10 clients (out of 10)
# DEBUG flower 2022-04-23 19:33:56,194 | server.py:214 | evaluate_round received 10 results and 0 failures
# DEBUG flower 2022-04-23 19:33:56,195 | server.py:255 | fit_round: strategy sampled 5 clients (out of 10)
# DEBUG flower 2022-04-23 19:39:39,636 | server.py:264 | fit_round received 5 results and 0 failures
# DEBUG flower 2022-04-23 19:39:40,027 | server.py:205 | evaluate_round: strategy sampled 10 clients (out of 10)
# DEBUG flower 2022-04-23 19:39:56,975 | server.py:214 | evaluate_round received 10 results and 0 failures
# DEBUG flower 2022-04-23 19:39:56,975 | server.py:255 | fit_round: strategy sampled 5 clients (out of 10)
# DEBUG flower 2022-04-23 19:42:40,258 | server.py:264 | fit_round received 5 results and 0 failures
# DEBUG flower 2022-04-23 19:42:40,652 | server.py:205 | evaluate_round: strategy sampled 10 clients (out of 10)
# DEBUG flower 2022-04-23 19:42:57,386 | server.py:214 | evaluate_round received 10 results and 0 failures
# DEBUG flower 2022-04-23 19:42:57,387 | server.py:255 | fit_round: strategy sampled 5 clients (out of 10)
# DEBUG flower 2022-04-23 19:48:44,202 | server.py:264 | fit_round received 5 results and 0 failures


# DEBUG flower 2022-04-23 19:48:44,594 | server.py:205 | evaluate_round: strategy sampled 10 clients (out of 10)
# DEBUG flower 2022-04-23 19:49:01,993 | server.py:214 | evaluate_round received 10 results and 0 failures
# DEBUG flower 2022-04-23 19:49:01,994 | server.py:255 | fit_round: strategy sampled 5 clients (out of 10)
# DEBUG flower 2022-04-23 19:51:46,024 | server.py:264 | fit_round received 5 results and 0 failures
# DEBUG flower 2022-04-23 19:51:46,399 | server.py:205 | evaluate_round: strategy sampled 10 clients (out of 10)
# DEBUG flower 2022-04-23 19:52:03,447 | server.py:214 | evaluate_round received 10 results and 0 failures
# DEBUG flower 2022-04-23 19:52:03,448 | server.py:255 | fit_round: strategy sampled 5 clients (out of 10)
# DEBUG flower 2022-04-23 19:55:54,531 | server.py:264 | fit_round received 5 results and 0 failures
# DEBUG flower 2022-04-23 19:55:54,895 | server.py:205 | evaluate_round: strategy sampled 10 clients (out of 10)
# DEBUG flower 2022-04-23 19:56:12,265 | server.py:214 | evaluate_round received 10 results and 0 failures
# DEBUG flower 2022-04-23 19:56:12,266 | server.py:255 | fit_round: strategy sampled 5 clients (out of 10)
# DEBUG flower 2022-04-23 19:58:51,467 | server.py:264 | fit_round received 5 results and 0 failures
# DEBUG flower 2022-04-23 19:58:51,903 | server.py:205 | evaluate_round: strategy sampled 10 clients (out of 10)
# DEBUG flower 2022-04-23 19:59:11,867 | server.py:214 | evaluate_round received 10 results and 0 failures
# DEBUG flower 2022-04-23 19:59:11,868 | server.py:255 | fit_round: strategy sampled 5 clients (out of 10)
# DEBUG flower 2022-04-23 20:01:45,606 | server.py:264 | fit_round received 5 results and 0 failures
# DEBUG flower 2022-04-23 20:01:46,008 | server.py:205 | evaluate_round: strategy sampled 10 clients (out of 10)
# DEBUG flower 2022-04-23 20:02:03,971 | server.py:214 | evaluate_round received 10 results and 0 failures
# DEBUG flower 2022-04-23 20:02:03,972 | server.py:255 | fit_round: strategy sampled 5 clients (out of 10)
# DEBUG flower 2022-04-23 20:04:46,399 | server.py:264 | fit_round received 5 results and 0 failures
# DEBUG flower 2022-04-23 20:04:46,798 | server.py:205 | evaluate_round: strategy sampled 10 clients (out of 10)
# DEBUG flower 2022-04-23 20:05:03,657 | server.py:214 | evaluate_round received 10 results and 0 failures
# DEBUG flower 2022-04-23 20:05:03,657 | server.py:255 | fit_round: strategy sampled 5 clients (out of 10)
# DEBUG flower 2022-04-23 20:07:40,023 | server.py:264 | fit_round received 5 results and 0 failures
# DEBUG flower 2022-04-23 20:07:40,404 | server.py:205 | evaluate_round: strategy sampled 10 clients (out of 10)
# DEBUG flower 2022-04-23 20:07:56,935 | server.py:214 | evaluate_round received 10 results and 0 failures
# DEBUG flower 2022-04-23 20:07:56,936 | server.py:255 | fit_round: strategy sampled 5 clients (out of 10)
# DEBUG flower 2022-04-23 20:13:42,197 | server.py:264 | fit_round received 5 results and 0 failures
# DEBUG flower 2022-04-23 20:13:42,578 | server.py:205 | evaluate_round: strategy sampled 10 clients (out of 10)
# DEBUG flower 2022-04-23 20:13:59,652 | server.py:214 | evaluate_round received 10 results and 0 failures
# DEBUG flower 2022-04-23 20:13:59,653 | server.py:255 | fit_round: strategy sampled 5 clients (out of 10)
# DEBUG flower 2022-04-23 20:17:00,792 | server.py:264 | fit_round received 5 results and 0 failures
# DEBUG flower 2022-04-23 20:17:01,159 | server.py:205 | evaluate_round: strategy sampled 10 clients (out of 10)
# DEBUG flower 2022-04-23 20:17:24,409 | server.py:214 | evaluate_round received 10 results and 0 failures
# DEBUG flower 2022-04-23 20:17:24,409 | server.py:255 | fit_round: strategy sampled 5 clients (out of 10)
# DEBUG flower 2022-04-23 20:23:59,982 | server.py:264 | fit_round received 5 results and 0 failures


# DEBUG flower 2022-04-23 20:24:00,347 | server.py:205 | evaluate_round: strategy sampled 10 clients (out of 10)
# DEBUG flower 2022-04-23 20:24:21,787 | server.py:214 | evaluate_round received 10 results and 0 failures
# DEBUG flower 2022-04-23 20:24:21,788 | server.py:255 | fit_round: strategy sampled 5 clients (out of 10)
# DEBUG flower 2022-04-23 20:30:58,730 | server.py:264 | fit_round received 5 results and 0 failures
# DEBUG flower 2022-04-23 20:30:59,092 | server.py:205 | evaluate_round: strategy sampled 10 clients (out of 10)
# DEBUG flower 2022-04-23 20:31:22,420 | server.py:214 | evaluate_round received 10 results and 0 failures
# DEBUG flower 2022-04-23 20:31:22,421 | server.py:255 | fit_round: strategy sampled 5 clients (out of 10)
# DEBUG flower 2022-04-23 20:33:59,418 | server.py:264 | fit_round received 5 results and 0 failures
# DEBUG flower 2022-04-23 20:33:59,743 | server.py:205 | evaluate_round: strategy sampled 10 clients (out of 10)
# DEBUG flower 2022-04-23 20:34:21,464 | server.py:214 | evaluate_round received 10 results and 0 failures
# DEBUG flower 2022-04-23 20:34:21,465 | server.py:255 | fit_round: strategy sampled 5 clients (out of 10)
# DEBUG flower 2022-04-23 20:37:15,570 | server.py:264 | fit_round received 5 results and 0 failures
# DEBUG flower 2022-04-23 20:37:15,935 | server.py:205 | evaluate_round: strategy sampled 10 clients (out of 10)
# DEBUG flower 2022-04-23 20:37:34,978 | server.py:214 | evaluate_round received 10 results and 0 failures
# DEBUG flower 2022-04-23 20:37:34,979 | server.py:255 | fit_round: strategy sampled 5 clients (out of 10)
# DEBUG flower 2022-04-23 20:41:58,858 | server.py:264 | fit_round received 5 results and 0 failures
# DEBUG flower 2022-04-23 20:41:59,216 | server.py:205 | evaluate_round: strategy sampled 10 clients (out of 10)
# DEBUG flower 2022-04-23 20:42:21,542 | server.py:214 | evaluate_round received 10 results and 0 failures
# DEBUG flower 2022-04-23 20:42:21,549 | server.py:255 | fit_round: strategy sampled 5 clients (out of 10)
# DEBUG flower 2022-04-23 20:48:56,272 | server.py:264 | fit_round received 5 results and 0 failures
# DEBUG flower 2022-04-23 20:48:56,665 | server.py:205 | evaluate_round: strategy sampled 10 clients (out of 10)
# DEBUG flower 2022-04-23 20:49:15,989 | server.py:214 | evaluate_round received 10 results and 0 failures
# DEBUG flower 2022-04-23 20:49:15,990 | server.py:255 | fit_round: strategy sampled 5 clients (out of 10)
# DEBUG flower 2022-04-23 20:52:13,601 | server.py:264 | fit_round received 5 results and 0 failures
# DEBUG flower 2022-04-23 20:52:14,011 | server.py:205 | evaluate_round: strategy sampled 10 clients (out of 10)
# DEBUG flower 2022-04-23 20:52:33,592 | server.py:214 | evaluate_round received 10 results and 0 failures
# DEBUG flower 2022-04-23 20:52:33,592 | server.py:255 | fit_round: strategy sampled 5 clients (out of 10)
# DEBUG flower 2022-04-23 20:56:54,558 | server.py:264 | fit_round received 5 results and 0 failures
# DEBUG flower 2022-04-23 20:56:54,939 | server.py:205 | evaluate_round: strategy sampled 10 clients (out of 10)
# DEBUG flower 2022-04-23 20:57:16,946 | server.py:214 | evaluate_round received 10 results and 0 failures
# DEBUG flower 2022-04-23 20:57:16,946 | server.py:255 | fit_round: strategy sampled 5 clients (out of 10)
# DEBUG flower 2022-04-23 21:01:37,836 | server.py:264 | fit_round received 5 results and 0 failures
# DEBUG flower 2022-04-23 21:01:38,198 | server.py:205 | evaluate_round: strategy sampled 10 clients (out of 10)
# DEBUG flower 2022-04-23 21:02:00,899 | server.py:214 | evaluate_round received 10 results and 0 failures
# DEBUG flower 2022-04-23 21:02:00,899 | server.py:255 | fit_round: strategy sampled 5 clients (out of 10)
# DEBUG flower 2022-04-23 21:09:36,715 | server.py:264 | fit_round received 5 results and 0 failures
# DEBUG flower 2022-04-23 21:09:37,154 | server.py:205 | evaluate_round: strategy sampled 10 clients (out of 10)
# DEBUG flower 2022-04-23 21:10:03,636 | server.py:214 | evaluate_round received 10 results and 0 failures
# DEBUG flower 2022-04-23 21:10:03,636 | server.py:255 | fit_round: strategy sampled 5 clients (out of 10)
# DEBUG flower 2022-04-23 21:14:56,656 | server.py:264 | fit_round received 5 results and 0 failures


# DEBUG flower 2022-04-23 21:14:57,098 | server.py:205 | evaluate_round: strategy sampled 10 clients (out of 10)
# DEBUG flower 2022-04-23 21:15:24,888 | server.py:214 | evaluate_round received 9 results and 1 failures
# DEBUG flower 2022-04-23 21:59:05,944 | server.py:255 | fit_round: strategy sampled 5 clients (out of 10)
# DEBUG flower 2022-04-23 22:02:20,466 | server.py:264 | fit_round received 4 results and 1 failures
# DEBUG flower 2022-04-23 22:02:20,778 | server.py:205 | evaluate_round: strategy sampled 10 clients (out of 10)
# DEBUG flower 2022-04-23 22:02:46,066 | server.py:214 | evaluate_round received 10 results and 0 failures
# DEBUG flower 2022-04-23 22:02:46,068 | server.py:255 | fit_round: strategy sampled 5 clients (out of 10)
# DEBUG flower 2022-04-23 22:07:35,650 | server.py:264 | fit_round received 5 results and 0 failures
# DEBUG flower 2022-04-23 22:07:36,032 | server.py:205 | evaluate_round: strategy sampled 10 clients (out of 10)
# DEBUG flower 2022-04-23 22:08:01,566 | server.py:214 | evaluate_round received 10 results and 0 failures
# DEBUG flower 2022-04-23 22:08:01,566 | server.py:255 | fit_round: strategy sampled 5 clients (out of 10)
# DEBUG flower 2022-04-23 22:11:34,113 | server.py:264 | fit_round received 5 results and 0 failures
# DEBUG flower 2022-04-23 22:11:34,529 | server.py:205 | evaluate_round: strategy sampled 10 clients (out of 10)
# DEBUG flower 2022-04-23 22:11:55,895 | server.py:214 | evaluate_round received 10 results and 0 failures
# DEBUG flower 2022-04-23 22:11:55,896 | server.py:255 | fit_round: strategy sampled 5 clients (out of 10)
# DEBUG flower 2022-04-23 22:16:50,357 | server.py:264 | fit_round received 5 results and 0 failures
# DEBUG flower 2022-04-23 22:16:50,740 | server.py:205 | evaluate_round: strategy sampled 10 clients (out of 10)
# DEBUG flower 2022-04-23 22:17:17,760 | server.py:214 | evaluate_round received 10 results and 0 failures
# DEBUG flower 2022-04-23 22:17:17,761 | server.py:255 | fit_round: strategy sampled 5 clients (out of 10)
# DEBUG flower 2022-04-23 22:20:33,076 | server.py:264 | fit_round received 5 results and 0 failures
# DEBUG flower 2022-04-23 22:20:33,464 | server.py:205 | evaluate_round: strategy sampled 10 clients (out of 10)
# DEBUG flower 2022-04-23 22:21:00,870 | server.py:214 | evaluate_round received 10 results and 0 failures
# DEBUG flower 2022-04-23 22:21:00,871 | server.py:255 | fit_round: strategy sampled 5 clients (out of 10)
# DEBUG flower 2022-04-23 22:23:57,758 | server.py:264 | fit_round received 5 results and 0 failures
# DEBUG flower 2022-04-23 22:23:58,146 | server.py:205 | evaluate_round: strategy sampled 10 clients (out of 10)
# DEBUG flower 2022-04-23 22:24:22,130 | server.py:214 | evaluate_round received 10 results and 0 failures
# DEBUG flower 2022-04-23 22:24:22,130 | server.py:255 | fit_round: strategy sampled 5 clients (out of 10)
# DEBUG flower 2022-04-23 22:27:13,468 | server.py:264 | fit_round received 5 results and 0 failures
# DEBUG flower 2022-04-23 22:27:13,845 | server.py:205 | evaluate_round: strategy sampled 10 clients (out of 10)
# DEBUG flower 2022-04-23 22:27:41,637 | server.py:214 | evaluate_round received 10 results and 0 failures
# DEBUG flower 2022-04-23 22:27:41,637 | server.py:255 | fit_round: strategy sampled 5 clients (out of 10)
# DEBUG flower 2022-04-23 22:35:01,008 | server.py:264 | fit_round received 5 results and 0 failures
# DEBUG flower 2022-04-23 22:35:01,414 | server.py:205 | evaluate_round: strategy sampled 10 clients (out of 10)
# DEBUG flower 2022-04-23 22:35:26,117 | server.py:214 | evaluate_round received 10 results and 0 failures
# DEBUG flower 2022-04-23 22:35:26,118 | server.py:255 | fit_round: strategy sampled 5 clients (out of 10)
# DEBUG flower 2022-04-23 22:43:47,941 | server.py:264 | fit_round received 5 results and 0 failures
# DEBUG flower 2022-04-23 22:43:48,275 | server.py:205 | evaluate_round: strategy sampled 10 clients (out of 10)
# DEBUG flower 2022-04-23 22:44:16,561 | server.py:214 | evaluate_round received 10 results and 0 failures
# DEBUG flower 2022-04-23 22:44:16,561 | server.py:255 | fit_round: strategy sampled 5 clients (out of 10)
# DEBUG flower 2022-04-23 22:48:27,025 | server.py:264 | fit_round received 5 results and 0 failures


# DEBUG flower 2022-04-23 22:48:27,431 | server.py:205 | evaluate_round: strategy sampled 10 clients (out of 10)
# DEBUG flower 2022-04-23 22:48:55,599 | server.py:214 | evaluate_round received 10 results and 0 failures
# DEBUG flower 2022-04-23 22:48:55,600 | server.py:255 | fit_round: strategy sampled 5 clients (out of 10)
# DEBUG flower 2022-04-23 22:52:48,443 | server.py:264 | fit_round received 5 results and 0 failures
# DEBUG flower 2022-04-23 22:52:48,815 | server.py:205 | evaluate_round: strategy sampled 10 clients (out of 10)
# DEBUG flower 2022-04-23 22:53:22,214 | server.py:214 | evaluate_round received 10 results and 0 failures
# DEBUG flower 2022-04-23 22:53:22,214 | server.py:255 | fit_round: strategy sampled 5 clients (out of 10)
# DEBUG flower 2022-04-23 23:01:51,183 | server.py:264 | fit_round received 5 results and 0 failures
# DEBUG flower 2022-04-23 23:01:51,546 | server.py:205 | evaluate_round: strategy sampled 10 clients (out of 10)
# DEBUG flower 2022-04-23 23:02:22,548 | server.py:214 | evaluate_round received 10 results and 0 failures
# DEBUG flower 2022-04-23 23:02:22,549 | server.py:255 | fit_round: strategy sampled 5 clients (out of 10)
# DEBUG flower 2022-04-23 23:09:43,734 | server.py:264 | fit_round received 5 results and 0 failures
# DEBUG flower 2022-04-23 23:09:44,074 | server.py:205 | evaluate_round: strategy sampled 10 clients (out of 10)
# DEBUG flower 2022-04-23 23:10:12,742 | server.py:214 | evaluate_round received 10 results and 0 failures
# DEBUG flower 2022-04-23 23:10:12,742 | server.py:255 | fit_round: strategy sampled 5 clients (out of 10)
# DEBUG flower 2022-04-23 23:13:52,484 | server.py:264 | fit_round received 5 results and 0 failures
# DEBUG flower 2022-04-23 23:13:52,875 | server.py:205 | evaluate_round: strategy sampled 10 clients (out of 10)
# DEBUG flower 2022-04-23 23:14:22,709 | server.py:214 | evaluate_round received 10 results and 0 failures
# DEBUG flower 2022-04-23 23:14:22,709 | server.py:255 | fit_round: strategy sampled 5 clients (out of 10)
# DEBUG flower 2022-04-23 23:18:11,120 | server.py:264 | fit_round received 5 results and 0 failures
# DEBUG flower 2022-04-23 23:18:11,523 | server.py:205 | evaluate_round: strategy sampled 10 clients (out of 10)
# DEBUG flower 2022-04-23 23:18:44,579 | server.py:214 | evaluate_round received 10 results and 0 failures
# DEBUG flower 2022-04-23 23:18:44,580 | server.py:255 | fit_round: strategy sampled 5 clients (out of 10)
# DEBUG flower 2022-04-23 23:22:22,170 | server.py:264 | fit_round received 5 results and 0 failures
# DEBUG flower 2022-04-23 23:22:22,537 | server.py:205 | evaluate_round: strategy sampled 10 clients (out of 10)
# DEBUG flower 2022-04-23 23:22:52,700 | server.py:214 | evaluate_round received 10 results and 0 failures
# DEBUG flower 2022-04-23 23:22:52,700 | server.py:255 | fit_round: strategy sampled 5 clients (out of 10)
# DEBUG flower 2022-04-23 23:31:14,564 | server.py:264 | fit_round received 5 results and 0 failures
# DEBUG flower 2022-04-23 23:31:14,954 | server.py:205 | evaluate_round: strategy sampled 10 clients (out of 10)
# DEBUG flower 2022-04-23 23:31:39,553 | server.py:214 | evaluate_round received 10 results and 0 failures
# DEBUG flower 2022-04-23 23:31:39,553 | server.py:255 | fit_round: strategy sampled 5 clients (out of 10)
# DEBUG flower 2022-04-23 23:39:54,427 | server.py:264 | fit_round received 5 results and 0 failures


# DEBUG flower 2022-04-23 23:39:54,793 | server.py:205 | evaluate_round: strategy sampled 10 clients (out of 10)
# DEBUG flower 2022-04-23 23:40:21,118 | server.py:214 | evaluate_round received 10 results and 0 failures
# DEBUG flower 2022-04-23 23:40:21,119 | server.py:255 | fit_round: strategy sampled 5 clients (out of 10)
# DEBUG flower 2022-04-23 23:46:31,131 | server.py:264 | fit_round received 5 results and 0 failures
# DEBUG flower 2022-04-23 23:46:31,504 | server.py:205 | evaluate_round: strategy sampled 10 clients (out of 10)
# DEBUG flower 2022-04-23 23:47:03,739 | server.py:214 | evaluate_round received 10 results and 0 failures
# DEBUG flower 2022-04-23 23:47:03,740 | server.py:255 | fit_round: strategy sampled 5 clients (out of 10)
# DEBUG flower 2022-04-23 23:51:09,118 | server.py:264 | fit_round received 5 results and 0 failures
# DEBUG flower 2022-04-23 23:51:09,500 | server.py:205 | evaluate_round: strategy sampled 10 clients (out of 10)
# DEBUG flower 2022-04-23 23:51:44,433 | server.py:214 | evaluate_round received 10 results and 0 failures
# DEBUG flower 2022-04-23 23:51:44,434 | server.py:255 | fit_round: strategy sampled 5 clients (out of 10)
# DEBUG flower 2022-04-23 23:55:47,343 | server.py:264 | fit_round received 5 results and 0 failures
# DEBUG flower 2022-04-23 23:55:47,716 | server.py:205 | evaluate_round: strategy sampled 10 clients (out of 10)
# DEBUG flower 2022-04-23 23:56:18,915 | server.py:214 | evaluate_round received 10 results and 0 failures
# DEBUG flower 2022-04-23 23:56:18,915 | server.py:255 | fit_round: strategy sampled 5 clients (out of 10)
# DEBUG flower 2022-04-24 00:00:21,373 | server.py:264 | fit_round received 5 results and 0 failures
# DEBUG flower 2022-04-24 00:00:21,765 | server.py:205 | evaluate_round: strategy sampled 10 clients (out of 10)
# DEBUG flower 2022-04-24 00:00:53,882 | server.py:214 | evaluate_round received 10 results and 0 failures
# DEBUG flower 2022-04-24 00:00:53,883 | server.py:255 | fit_round: strategy sampled 5 clients (out of 10)
# DEBUG flower 2022-04-24 00:04:55,318 | server.py:264 | fit_round received 5 results and 0 failures
# DEBUG flower 2022-04-24 00:04:55,721 | server.py:205 | evaluate_round: strategy sampled 10 clients (out of 10)
# DEBUG flower 2022-04-24 00:05:27,399 | server.py:214 | evaluate_round received 10 results and 0 failures
# DEBUG flower 2022-04-24 00:05:27,400 | server.py:255 | fit_round: strategy sampled 5 clients (out of 10)
# DEBUG flower 2022-04-24 00:14:40,037 | server.py:264 | fit_round received 5 results and 0 failures
# DEBUG flower 2022-04-24 00:14:40,430 | server.py:205 | evaluate_round: strategy sampled 10 clients (out of 10)
# DEBUG flower 2022-04-24 00:15:14,027 | server.py:214 | evaluate_round received 10 results and 0 failures
# DEBUG flower 2022-04-24 00:15:14,027 | server.py:255 | fit_round: strategy sampled 5 clients (out of 10)
# DEBUG flower 2022-04-24 00:19:15,919 | server.py:264 | fit_round received 5 results and 0 failures
# DEBUG flower 2022-04-24 00:19:16,298 | server.py:205 | evaluate_round: strategy sampled 10 clients (out of 10)
# DEBUG flower 2022-04-24 00:19:49,878 | server.py:214 | evaluate_round received 10 results and 0 failures
# DEBUG flower 2022-04-24 00:19:49,879 | server.py:255 | fit_round: strategy sampled 5 clients (out of 10)
# DEBUG flower 2022-04-24 00:25:58,959 | server.py:264 | fit_round received 5 results and 0 failures
# DEBUG flower 2022-04-24 00:25:59,386 | server.py:205 | evaluate_round: strategy sampled 10 clients (out of 10)
# DEBUG flower 2022-04-24 00:26:37,243 | server.py:214 | evaluate_round received 10 results and 0 failures
# DEBUG flower 2022-04-24 00:26:37,243 | server.py:255 | fit_round: strategy sampled 5 clients (out of 10)
# DEBUG flower 2022-04-24 00:30:39,784 | server.py:264 | fit_round received 5 results and 0 failures
# DEBUG flower 2022-04-24 00:30:40,140 | server.py:205 | evaluate_round: strategy sampled 10 clients (out of 10)
# DEBUG flower 2022-04-24 00:31:15,914 | server.py:214 | evaluate_round received 10 results and 0 failures
# DEBUG flower 2022-04-24 00:31:15,914 | server.py:255 | fit_round: strategy sampled 5 clients (out of 10)
# DEBUG flower 2022-04-24 00:35:19,703 | server.py:264 | fit_round received 5 results and 0 failures


# DEBUG flower 2022-04-24 00:35:20,114 | server.py:205 | evaluate_round: strategy sampled 10 clients (out of 10)
# DEBUG flower 2022-04-24 00:35:51,167 | server.py:214 | evaluate_round received 10 results and 0 failures
# DEBUG flower 2022-04-24 00:35:51,168 | server.py:255 | fit_round: strategy sampled 5 clients (out of 10)
# DEBUG flower 2022-04-24 00:44:16,395 | server.py:264 | fit_round received 3 results and 2 failures
# DEBUG flower 2022-04-24 01:19:22,215 | server.py:205 | evaluate_round: strategy sampled 10 clients (out of 10)
# DEBUG flower 2022-04-24 01:19:53,647 | server.py:214 | evaluate_round received 10 results and 0 failures
# DEBUG flower 2022-04-24 01:19:53,647 | server.py:255 | fit_round: strategy sampled 5 clients (out of 10)
# DEBUG flower 2022-04-24 01:24:27,303 | server.py:264 | fit_round received 5 results and 0 failures
# DEBUG flower 2022-04-24 01:24:27,661 | server.py:205 | evaluate_round: strategy sampled 10 clients (out of 10)
# DEBUG flower 2022-04-24 01:25:00,558 | server.py:214 | evaluate_round received 10 results and 0 failures
# DEBUG flower 2022-04-24 01:25:00,558 | server.py:255 | fit_round: strategy sampled 5 clients (out of 10)
# DEBUG flower 2022-04-24 01:29:48,143 | server.py:264 | fit_round received 5 results and 0 failures
# DEBUG flower 2022-04-24 01:29:48,520 | server.py:205 | evaluate_round: strategy sampled 10 clients (out of 10)
# DEBUG flower 2022-04-24 01:30:19,637 | server.py:214 | evaluate_round received 10 results and 0 failures
# DEBUG flower 2022-04-24 01:30:19,638 | server.py:255 | fit_round: strategy sampled 5 clients (out of 10)
# DEBUG flower 2022-04-24 01:34:50,951 | server.py:264 | fit_round received 5 results and 0 failures
# DEBUG flower 2022-04-24 01:34:51,343 | server.py:205 | evaluate_round: strategy sampled 10 clients (out of 10)
# DEBUG flower 2022-04-24 01:35:26,492 | server.py:214 | evaluate_round received 10 results and 0 failures
# DEBUG flower 2022-04-24 01:35:26,493 | server.py:255 | fit_round: strategy sampled 5 clients (out of 10)
# DEBUG flower 2022-04-24 01:40:15,508 | server.py:264 | fit_round received 5 results and 0 failures
# DEBUG flower 2022-04-24 01:40:15,918 | server.py:205 | evaluate_round: strategy sampled 10 clients (out of 10)
# DEBUG flower 2022-04-24 01:40:52,772 | server.py:214 | evaluate_round received 10 results and 0 failures
# DEBUG flower 2022-04-24 01:40:52,773 | server.py:255 | fit_round: strategy sampled 5 clients (out of 10)
# DEBUG flower 2022-04-24 01:45:20,168 | server.py:264 | fit_round received 5 results and 0 failures
# DEBUG flower 2022-04-24 01:45:20,547 | server.py:205 | evaluate_round: strategy sampled 10 clients (out of 10)
# DEBUG flower 2022-04-24 01:45:53,481 | server.py:214 | evaluate_round received 10 results and 0 failures
# DEBUG flower 2022-04-24 01:45:53,482 | server.py:255 | fit_round: strategy sampled 5 clients (out of 10)
# DEBUG flower 2022-04-24 01:52:36,414 | server.py:264 | fit_round received 5 results and 0 failures
# DEBUG flower 2022-04-24 01:52:36,816 | server.py:205 | evaluate_round: strategy sampled 10 clients (out of 10)
# DEBUG flower 2022-04-24 01:53:10,258 | server.py:214 | evaluate_round received 10 results and 0 failures
# DEBUG flower 2022-04-24 01:53:10,259 | server.py:255 | fit_round: strategy sampled 5 clients (out of 10)
# DEBUG flower 2022-04-24 01:57:42,057 | server.py:264 | fit_round received 5 results and 0 failures
# DEBUG flower 2022-04-24 01:57:42,428 | server.py:205 | evaluate_round: strategy sampled 10 clients (out of 10)
# DEBUG flower 2022-04-24 01:58:21,409 | server.py:214 | evaluate_round received 10 results and 0 failures
# DEBUG flower 2022-04-24 01:58:21,409 | server.py:255 | fit_round: strategy sampled 5 clients (out of 10)
# DEBUG flower 2022-04-24 02:02:47,943 | server.py:264 | fit_round received 5 results and 0 failures


# DEBUG flower 2022-04-24 02:02:48,337 | server.py:205 | evaluate_round: strategy sampled 10 clients (out of 10)
# DEBUG flower 2022-04-24 02:03:20,507 | server.py:214 | evaluate_round received 10 results and 0 failures
# DEBUG flower 2022-04-24 02:03:20,507 | server.py:255 | fit_round: strategy sampled 5 clients (out of 10)
# DEBUG flower 2022-04-24 02:07:59,678 | server.py:264 | fit_round received 5 results and 0 failures
# DEBUG flower 2022-04-24 02:08:00,032 | server.py:205 | evaluate_round: strategy sampled 10 clients (out of 10)
# DEBUG flower 2022-04-24 02:08:41,025 | server.py:214 | evaluate_round received 10 results and 0 failures
# DEBUG flower 2022-04-24 02:08:41,026 | server.py:255 | fit_round: strategy sampled 5 clients (out of 10)
# DEBUG flower 2022-04-24 02:13:00,393 | server.py:264 | fit_round received 4 results and 1 failures
# DEBUG flower 2022-04-24 02:44:55,093 | server.py:205 | evaluate_round: strategy sampled 10 clients (out of 10)
# DEBUG flower 2022-04-24 02:46:06,760 | server.py:214 | evaluate_round received 10 results and 0 failures
# DEBUG flower 2022-04-24 02:46:06,760 | server.py:255 | fit_round: strategy sampled 5 clients (out of 10)
# DEBUG flower 2022-04-24 02:50:27,043 | server.py:264 | fit_round received 4 results and 1 failures
# DEBUG flower 2022-04-24 03:14:03,704 | server.py:205 | evaluate_round: strategy sampled 10 clients (out of 10)
# DEBUG flower 2022-04-24 03:15:04,204 | server.py:214 | evaluate_round received 10 results and 0 failures
# DEBUG flower 2022-04-24 03:15:04,204 | server.py:255 | fit_round: strategy sampled 5 clients (out of 10)
# DEBUG flower 2022-04-24 03:20:23,495 | server.py:264 | fit_round received 5 results and 0 failures
# DEBUG flower 2022-04-24 03:20:23,868 | server.py:205 | evaluate_round: strategy sampled 10 clients (out of 10)
# DEBUG flower 2022-04-24 03:21:22,924 | server.py:214 | evaluate_round received 10 results and 0 failures
# DEBUG flower 2022-04-24 03:21:22,925 | server.py:255 | fit_round: strategy sampled 5 clients (out of 10)
# DEBUG flower 2022-04-24 03:26:40,480 | server.py:264 | fit_round received 5 results and 0 failures
# DEBUG flower 2022-04-24 03:26:40,851 | server.py:205 | evaluate_round: strategy sampled 10 clients (out of 10)
# DEBUG flower 2022-04-24 03:27:45,075 | server.py:214 | evaluate_round received 10 results and 0 failures
# DEBUG flower 2022-04-24 03:27:45,076 | server.py:255 | fit_round: strategy sampled 5 clients (out of 10)
# DEBUG flower 2022-04-24 03:30:07,281 | server.py:264 | fit_round received 5 results and 0 failures
# DEBUG flower 2022-04-24 03:30:07,632 | server.py:205 | evaluate_round: strategy sampled 10 clients (out of 10)
# DEBUG flower 2022-04-24 03:30:55,108 | server.py:214 | evaluate_round received 10 results and 0 failures
# DEBUG flower 2022-04-24 03:30:55,109 | server.py:255 | fit_round: strategy sampled 5 clients (out of 10)
# DEBUG flower 2022-04-24 03:35:13,015 | server.py:264 | fit_round received 5 results and 0 failures
# DEBUG flower 2022-04-24 03:35:13,421 | server.py:205 | evaluate_round: strategy sampled 10 clients (out of 10)
# DEBUG flower 2022-04-24 03:36:11,681 | server.py:214 | evaluate_round received 10 results and 0 failures
# DEBUG flower 2022-04-24 03:36:11,682 | server.py:255 | fit_round: strategy sampled 5 clients (out of 10)
# DEBUG flower 2022-04-24 03:40:29,844 | server.py:264 | fit_round received 5 results and 0 failures
# DEBUG flower 2022-04-24 03:40:30,214 | server.py:205 | evaluate_round: strategy sampled 10 clients (out of 10)
# DEBUG flower 2022-04-24 03:41:18,543 | server.py:214 | evaluate_round received 10 results and 0 failures
# DEBUG flower 2022-04-24 03:41:18,543 | server.py:255 | fit_round: strategy sampled 5 clients (out of 10)
# DEBUG flower 2022-04-24 03:45:37,474 | server.py:264 | fit_round received 5 results and 0 failures
# DEBUG flower 2022-04-24 03:45:37,878 | server.py:205 | evaluate_round: strategy sampled 10 clients (out of 10)
# DEBUG flower 2022-04-24 03:46:25,286 | server.py:214 | evaluate_round received 10 results and 0 failures
# DEBUG flower 2022-04-24 03:46:25,287 | server.py:255 | fit_round: strategy sampled 5 clients (out of 10)
# DEBUG flower 2022-04-24 03:50:38,351 | server.py:264 | fit_round received 5 results and 0 failures
# DEBUG flower 2022-04-24 03:50:38,700 | server.py:205 | evaluate_round: strategy sampled 10 clients (out of 10)
# DEBUG flower 2022-04-24 03:51:45,202 | server.py:214 | evaluate_round received 10 results and 0 failures
# DEBUG flower 2022-04-24 03:51:45,203 | server.py:255 | fit_round: strategy sampled 5 clients (out of 10)
# DEBUG flower 2022-04-24 03:54:31,296 | server.py:264 | fit_round received 5 results and 0 failures
# DEBUG flower 2022-04-24 03:54:31,642 | server.py:205 | evaluate_round: strategy sampled 10 clients (out of 10)
# DEBUG flower 2022-04-24 03:55:16,928 | server.py:214 | evaluate_round received 10 results and 0 failures
# INFO flower 2022-04-24 03:55:16,928 | server.py:172 | FL finished in 34694.52637966722
# INFO flower 2022-04-24 03:55:16,935 | app.py:119 | app_fit: losses_distributed [(1, 0.47666667178273203), (2, 0.6616666639223695), (3, 0.7583333315327764), (4, 0.8650000035762787), (5, 0.8816666707396508), (6, 0.9008333414793015), (7, 0.9758333444595337), (8, 0.9775000035762786), (9, 0.9783333361148834), (10, 0.717083340883255), (11, 0.8212499916553497), (12, 0.870000010728836), (13, 0.9091666638851166), (14, 0.9208333492279053), (15, 0.9283333420753479), (16, 0.9362499952316284), (17, 0.9437500059604644), (18, 0.9466666758060456), (19, 0.9412500083446502), (20, 0.7991666674613953), (21, 0.8594444513320922), (22, 0.8816666781902314), (23, 0.8947222292423248), (24, 0.9180555641651154), (25, 0.9313889086246491), (26, 0.9269444406032562), (27, 0.9280555605888366), (28, 0.9372222304344178), (29, 0.9550000011920929), (30, 0.8479166567325592), (31, 0.8764583468437195), (32, 0.9399999976158142), (33, 0.9410416662693024), (34, 0.9477083444595337), (35, 0.9641666769981384), (36, 0.9637499988079071), (37, 0.9664583563804626), (38, 0.9647916913032532), (39, 0.9652083337306976), (40, 0.8358333230018615), (41, 0.8856666684150696), (42, 0.904833322763443), (43, 0.920499986410141), (44, 0.9263333022594452), (45, 0.9358333349227905), (46, 0.9306666672229766), (47, 0.9371666491031647), (48, 0.934333324432373), (49, 0.9423333048820496), (50, 0.8675000071525574), (51, 0.9086419807540046), (52, 0.8356944214552641), (53, 0.8712499961256981), (54, 0.870833332836628), (55, 0.8718055620789528), (56, 0.8725000068545341), (57, 0.8706944644451141), (58, 0.8755555793642997), (59, 0.8737499967217446), (60, 0.8120238065719605), (61, 0.8472619116306305), (62, 0.8613095372915268), (63, 0.8611904978752136), (64, 0.8716666847467422), (65, 0.8771428734064102), (66, 0.8804762125015259), (67, 0.8809523791074753), (68, 0.8820238143205643), (69, 0.8816666632890702), (70, 0.8390625), (71, 0.8713541746139526), (72, 0.8739583432674408), (73, 0.8809374988079071), (74, 0.8964583367109299), (75, 0.8991666674613953), (76, 0.8929166555404663), (77, 0.8980208545923233), (78, 0.9008333444595337), (79, 0.8955208390951157), (80, 0.677962988615036), (81, 0.7144444517791271), (82, 0.7343518741428852), (83, 0.7350925795733929), (84, 0.740740743279457), (85, 0.7480555504560471), (86, 0.7498148009181023), (87, 0.7478703737258912), (88, 0.7496296375989914), (89, 0.7519444569945335), (90, 0.6280833050608635), (91, 0.5639166578650474), (92, 0.5877500131726265), (93, 0.5906666822731494), (94, 0.5991666689515114), (95, 0.5951666697859764), (96, 0.6018333211541176), (97, 0.5974999874830246), (98, 0.6014166504144669), (99, 0.6001666590571404)]
# INFO flower 2022-04-24 03:55:16,935 | app.py:120 | app_fit: metrics_distributed {}
# INFO flower 2022-04-24 03:55:16,935 | app.py:121 | app_fit: losses_centralized []
# INFO flower 2022-04-24 03:55:16,935 | app.py:122 | app_fit: metrics_centralized {}
#
# Process finished with exit code 0