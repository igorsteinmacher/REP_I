#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ =  'Felipe Fronchetti'
__contact__ = 'fronchetti@usp.br'

import os
import github_scraper.scraper as scraper
import github_scraper.project_collector as project_collector_api

raw_files_folder = '../raw_files'
spreadsheet_folder = '../spreadsheets'

class DocumentationDownloader:
    def __init__(self, project_org, project_name, collector):
        self.project_org, self.project_name, self.project_collector = project_org, project_name, project_collector
        self.project_folder = os.path.join(raw_files_folder, self.project_org + '#' + self.project_name)
        excluded = []

        if os.path.isfile(os.path.join(raw_files_folder, 'excluded_projects.txt')):
            with open(os.path.join(raw_files_folder, 'excluded_projects.txt'), 'r') excluded_file:
                excluded = excluded_file.read().splitlines()

        if self.project_name not in self.exclude:
            self.__download_files()

    def __download_files(self):
        readme, readme_filename = project_collector.readme()
        contributing, contributing_filename = project_collector.contributing()

        if readme and contributing:
            if not os.path.isdir(self.project_folder):
                os.makedirs(self.project_folder)

            with open(os.path.join(self.project_folder, readme_filename), 'wb') as output_file:
                output_file.write(str.encode(readme, encoding='utf-8'))

            with open(os.path.join(self.project_folder, contributing_filename), 'wb') as output_file:
                output_file.write(str.encode(contributing, encoding='utf-8'))
        else:
            with open(os.path.join(raw_files_folder, 'excluded_projects.txt'), 'a') as output_file:
                output_file.write(self.project_org + '/' + self.project_name + '\n')

if __name__ == '__main__':
    api_client_id = '4161a8257efaea420c94'
    api_client_secret = 'd814ec48927a6bd62c55c058cd028a949e5362d4'
    api_scraper = scraper.Create(api_client_id, api_client_secret)

    projects = {
        1: [('alexreisner', 'geocoder'), ('atom', 'atom-shell'), ('bjorn', 'tiled'), ('bumptech', 'glide'), ('celery', 'celery'), ('celluloid', 'celluloid'),
            ('dropwizard', 'dropwizard'), ('dropwizard', 'metrics'), ('erikhuda', 'thor'), ('Eugeny', 'ajenti'), ('getsen-try', 'sentry'), ('github', 'android'),
            ('gruntjs', 'grunt'), ('janl', 'mustache.js'), ('jr-burke', 'requirejs'), ('justinfrench', 'formtastic'), ('kivy', 'kivy'), ('koush', 'ion'),
            ('kriswallsmith', 'assetic'), ('Leaflet', 'Leaflet'), ('less', 'less.js'), ('mailpile', 'Mailpile'), ('mbostock', 'd3'), ('mitchellh', 'vagrant'),
            ('mitsuhiko', 'flask'), ('mongoid', 'mongoid'), ('nate-parrott', 'Flashlight'), ('nicolasgramlich', 'AndEngine'), ('paulas-muth', 'fnordmetric'),
            ('phacility', 'phabricator'), ('powerline', 'powerline'), ('puphpet', 'puphpet'), ('ratchetphp', 'Ratchet'), ('ReactiveX', 'RxJava'),
            ('sandstorm-io', 'capnproto'), ('sass', 'sass'), ('sebastianbergmann', 'phpunit'), ('sferik', 'twitter'), ('silexphp', 'Silex'),
            ('sstephenson', 'sprockets'), ('substack', 'node-browserify'), ('thoughtbot', 'factory_girl'), ('thoughtbot', 'paperclip'), ('wp-cli', 'wp-cli')],
        2: [('activeadmin', 'activeadmin'), ('ajaxorg', 'ace'), ('ansible', 'ansible'), ('apache', 'cassandra'), ('bup', 'bup'), ('clojure', 'clojure'),
            ('composer', 'composer'), ('cucumber', 'cucumber'), ('driftyco', 'ionic'), ('drupal', 'drupal'), ('elas-ticsearch', 'elasticsearch'),
            ('elasticsearch', 'logstash'), ('ex-cilys', 'androidannotations'), ('facebook', 'osquery'), ('facebook', 'presto'), ('FriendsOfPHP', 'PHP-CS-Fixer'),
            ('github', 'linguist'), ('Itseez', 'opencv'), ('jadejs', 'jade'), ('jashkenas', 'backbone'), ('JohnLangford', 'vowpal_wabbit'), ('jquery', 'jquery-ui'),
            ('libgdx', 'libgdx'), ('meskyanichi', 'backup'), ('netty', 'netty'), ('omab', 'django-social-auth'), ('openframeworks', 'openFrameworks'),
            ('plataformatec', 'devise'), ('prawnpdf', 'prawn'), ('pydata', 'pandas'), ('Re-spect', 'Validation'), ('sampsyo', 'beets'), ('SFTtech', 'openage'),
            ('sparklemo-tion', 'nokogiri'), ('strongloop', 'express'), ('thinkaurelius', 'titan'), ('ThinkU-pLLC', 'ThinkUp'), ('thumbor', 'thumbor'),
            ('xetorthio', 'jedis')],
        3: [('bbatsov', 'rubocop'), ('bitcoin', 'bitcoin'), ('bundler', 'bundler'), ('divio', 'django-cms'), ('haml', 'haml'), ('jnicklas', 'capybara'),
            ('mozilla', 'pdf.js'), ('rg3', 'youtube-dl'), ('mrdoob', 'three.js'), ('springprojects', 'spring-framework'), ('yiisoft', 'yii2')],
        4: [('boto', 'boto'), ('BVLC', 'caffe'), ('codemirror', 'CodeMirror'), ('gradle', 'gradle'), ('ipython', 'ipython'), ('jekyll', 'jekyll'), ('jquery', 'jquery')],
        5: [('iojs', 'io.js'), ('meteor', 'meteor'), ('ruby', 'ruby'), ('WordPress', 'WordPress')],
        6: [('chef', 'chef'), ('cocos2d', 'cocos2d-x'), ('diaspora', 'diaspora'), ('emberjs', 'ember.js'), ('resque', 'resque'), ('Shopify', 'active_merchant'), ('spotify', 'luigi'), ('TryGhost', 'Ghost')],
        7: [('django', 'django'), ('joomla', 'joomla-cms'), ('scikit-learn', 'scikit-learn')],
        9: [('JetBrains', 'intellij-community'), ('puppetlabs', 'puppet'), ('rails', 'rails')],
        11: [('saltstack', 'salt'), ('Seldaek', 'monolog'), ('v8', 'v8')],
        12: [('git', 'git'), ('webscalesql', 'webscalesql-5.6')],
        13: [('fog', 'fog')],
        14: [('odoo', 'odoo')]
    }

    if not os.path.isdir(raw_files_folder):
        os.mkdir(raw_files_folder)

    if not os.path.isdir(spreadsheet_folder):
        os.mkdir(spreadsheet_folder)

    for truck_factor in projects:
        for project in projects[truck_factor]:
            project_name = project[1]
            project_org = project[0]
            project_collector = project_collector_api.Collector(project_org, project_name, api_scraper)
            creator = DocumentationDownloader(project_org, project_name, project_collector)