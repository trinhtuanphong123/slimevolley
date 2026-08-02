from setuptools import setup

setup(
    name='slimevolleygym',
    version='0.2.0',
    keywords='games, environment, agent, rl, ai, gymnasium',
    url='https://github.com/hardmaru/slimevolleygym',
    description='Slime Volleyball Gymnasium Environment',
    packages=['slimevolleygym'],
    python_requires='>=3.9',
    install_requires=[
        'gymnasium>=1.3.0',
        'numpy>=1.24',
        'opencv-python>=4.0',
        'pyglet>=1.5,<2.0',
    ],
    extras_require={
        # install with: pip install -e .[training]
        'training': [
            'stable-baselines3>=2.9.0',
            'torch>=2.0',
            'tensorboard>=2.9',
        ],
    },
)
