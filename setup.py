from setuptools import setup

setup(
    name='Discretely-Indexed-Flows',
    version='',
    packages=['models'],
    url='',
    license='',
    author='EA264728',
    author_email='elouan.argouarch@gmail.com',
    setup_requires = ['wheel'],
    install_requires = ['torch','tqdm','jupyter','matplotlib', 'seaborn',
                        'ExpectationMaximisation @ git+https://github.com/ElouanARGOUARCH/ExpectationMaximisation.git',
                        "TargetExamples @ git+https://github.com/ElouanARGOUARCH/TargetExamples.git"],
    description=''
)
