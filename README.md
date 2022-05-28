<p align="center">
  <img src="assets/amgu.png" height="150" />
</p>

[![CI/CD](https://github.com/dev0Guy/amgu/actions/workflows/python-package-conda.yml/badge.svg)](https://github.com/dev0Guy/amgu/actions/workflows/python-package-conda.yml)
[![format](https://github.com/dev0Guy/amgu/actions/workflows/black.yml/badge.svg)](https://github.com/dev0Guy/amgu/actions/workflows/black.yml)

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">s</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>

<!-- ABOUT THE PROJECT -->
## About Amgu

The majority of paper in the subject of DRL & traffic mannagment don't share thier code or the
data they used for train.
There is an need to organize & create uniforms between paper &code a like.
Amgu try to execly do that, creating abstract classes that can be hierarchical from
and be easy to use and prevent cooled start. Traffic Managment System (including agent) has been created using thoes classes.Amgu-Traffic enable developer to use easy API with custom & premade
model and envierments & Attacks on thoes.

<p align="center">
  <img src="assets/cityflow.gif" height="350" />
</p>


### Built With

Amgu has been built using, the following:

* [Python](https://www.python.org/)
* [PyPi](https://pypi.org/)
* [Cityflow](https://github.com/cityflow-project/CityFlow)
* [Ray-rllib](https://github.com/ray-project/ray/blob/master/python/ray/rllib)

<!-- GETTING STARTED -->
## Getting Started

To use this project you'll need to have CityFlow already install inside your pip enviorment.
In Addition ray should be installed custom to your machine(x86/x64).

### Installation

Bellow example show the step to install and run example of Amgu.
1. Install Amgu Abstract using Pypi.
  ```sh
      pip install amgu/amgu_abstract/.
   ```
2. Install Amgu Traffic using Pypi.
   ```sh
      pip install amgu/amgu_traffic/.
   ```
3. Import to your project.
   ```python
    import amgu_abstract
    import amgu_traffic
   ```
4. Use in your code.
   ```python
      from amgu_traffic import DiscreteCF, AvgWaitingTime, CNN, RayRunner, Vanila
      config = {
          "env_config": {
              "config_path": "examples/hangzhou_1x1_bc-tyc_18041607_1h/config.json",
              "steps_per_episode": 100,
              "res_path": "res/",
          },
          "stop": {"training_iteration": 5},
          "res_path": "res/",
          "framework": "torch",
          "seed": 123,
          "evaluation_interval": 10,
          "evaluation_duration": 5,
          "exploration_config": {
              "type": "EpsilonGreedy",
              "epsilon_schedule": {
                  "type": "ExponentialSchedule",
                  "initial_p": 1,
                  "schedule_timesteps": 100 // 5,
                  "decay_rate": 0.99,
              },
          },
          "model": {
              "custom_model": "new_models",
              "custom_model_config": {
                  "intersection_num": 1,
                  "hidden_size": 10,
              },
          },
          "run_from": "/Users/guyarieli/Documents/GitHub/amgu/amgu/",
          "env": "custom_env",
      }
      preprocess_dict = {"func": Vanila, "argument_list": []}
      env_func = lambda _: DiscreteCF(config["env_config"], AvgWaitingTime, preprocess_dict)
      runner = RayRunner(config, CNN, env_func, "DQN")
      runner.train()

   ```

<!-- ROADMAP -->
## Roadmap
- [x] Abstract Class.
- [x] Traffic Class (using cityflow).
- [ ] Add Documntation.
- [x] Build Test

See the [open issues](https://github.com/dev0Guy/amgu/issues) for a full list of proposed features (and known issues).

<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE` for more information.

<p align="right">(<a href="#top">back to top</a>)</p>
