from testing.coarse_tester import CoarseTester
from testing.coarse_to_fine_tester import CoarseToFineTester
from common import config


# COARSE ONLY
if 1:
    #estimation_method = 'oracle'
    #estimation_method = 'current_image'
    #estimation_method = 'first_image'
    #estimation_method = 'best_image_dropout'
    #estimation_method = 'best_image_predicted'
    #estimation_method = 'batch'
    #estimation_method = 'batch_with_dropout_uncertainty'
    #estimation_method = 'batch_with_predicted_uncertainty'
    #estimation_method = 'filtering_with_predicted_uncertainty'
    #estimation_method = 'filtering_with_dropout_uncertainty'
    estimation_method = 'filtering_with_static_uncertainty'
    tester = CoarseTester()
    tester.run(task_name=config.TASK_NAME, estimation_method=estimation_method)

# COARSE TO FINE
if 0:
    estimation_method = 'filtering_with_static_uncertainty'
    use_correction = True
    tester = CoarseToFineTester()
    tester.run_episodes(task_name=config.TASK_NAME, estimation_method=estimation_method, use_correction=use_correction)
