import cityflow

eng = cityflow.Engine("examples/syn_1x1_uniform_2400_1h/config.json", thread_num=5)

action = 0
counter = 0
for i in range(5_000):
    eng.set_tl_phase('intersection_1_1', action)
    eng.next_step()
    counter += 1
    if counter == 100:
        counter = 0
        action = (action + 1) % 4