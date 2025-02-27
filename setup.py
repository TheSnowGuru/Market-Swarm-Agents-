from setuptools import setup, find_packages

setup(
    name='market_swarm_agents',
    version='0.1',
    packages=find_packages(),
    entry_points={
        'console_scripts': [
            'swarm=interactive_cli:main',
        ],
    },
    install_requires=[
        # Include key dependencies from requirements.txt
        'pandas>=2.1.2',
        'numpy>=1.26.0',
        'rich>=13.7.0',
        'questionary>=2.0.1',
    ],
    author='Market Swarm Team',
    description='AI-powered multi-agent trading platform',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/your-repo/market-swarm-agents',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
    ],
)
